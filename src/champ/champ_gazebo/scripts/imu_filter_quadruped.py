#!/usr/bin/env python3
"""
Quadruped-specialized IMU filter (ROS 2).

Improvements over a simple EMA filter:
  1. Contact-aware adaptive filtering  – boosts smoothing for N ms after each
     foot touchdown to suppress ground-impact spikes.
  2. Complementary filter for roll/pitch – blends gyro integration (short-term
     accuracy) with accelerometer tilt estimate (long-term drift correction).
  3. Gravity compensation output        – publishes gravity-free linear
     acceleration on imu/data/gravity_free (optional).
  4. Outlier / spike rejection          – discards samples that deviate too far
     from the running estimate (sensor glitches, shock artifacts).
  5. Adaptive measurement covariance    – inflates linear-acceleration and
     orientation covariance during impact windows so the downstream EKF
     automatically down-weights noisy measurements.

Topics
------
  Subscribed:
    imu/data_raw          sensor_msgs/Imu           raw IMU input
    foot_contacts         champ_msgs/ContactsStamped 4-leg contact flags

  Published:
    imu/data              sensor_msgs/Imu  filtered output (replaces raw)
    imu/data/gravity_free sensor_msgs/Imu  gravity-removed accel (optional)

Parameters
----------
  accel_alpha         float  EMA weight for linear acceleration   (0–1, def 0.30)
  gyro_alpha          float  EMA weight for angular velocity       (0–1, def 0.40)
  impact_alpha        float  EMA weight during foot-impact window  (0–1, def 0.08)
  impact_duration_ms  float  Impact-mode duration after touchdown  (ms,  def 100)
  comp_alpha          float  Gyro trust in complementary filter    (0–1, def 0.98)
  gravity             float  Local gravity magnitude               (m/s², def 9.81)
  outlier_accel       float  Accel spike rejection threshold       (m/s², def 30.0)
  outlier_gyro        float  Gyro  spike rejection threshold       (rad/s, def 10.0)
  use_contacts        bool   Enable contact-aware mode             (def True)
  publish_gravity_free bool  Publish gravity-free accel topic      (def False)
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

try:
    from champ_msgs.msg import ContactsStamped
    _HAS_CHAMP_MSGS = True
except ImportError:
    _HAS_CHAMP_MSGS = False


# ── Quaternion / Euler helpers ────────────────────────────────────────────────

def _accel_to_roll_pitch(ax: float, ay: float, az: float):
    """Estimate roll and pitch from accelerometer (assumes near-static body)."""
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 0.5:          # too small to give a reliable tilt estimate
        return 0.0, 0.0
    ax /= norm
    ay /= norm
    az /= norm
    roll  = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    return roll, pitch


def _euler_to_quat(roll: float, pitch: float, yaw: float):
    """Convert ZYX Euler angles to quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll  * 0.5), math.sin(roll  * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw   * 0.5), math.sin(yaw   * 0.5)
    w =  cr * cp * cy + sr * sp * sy
    x =  sr * cp * cy - cr * sp * sy
    y =  cr * sp * cy + sr * cp * sy
    z =  cr * cp * sy - sr * sp * cy
    return x, y, z, w


def _quat_yaw(q) -> float:
    """Extract yaw from a geometry_msgs/Quaternion-like object."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _gravity_body(roll: float, pitch: float, g: float):
    """Gravity vector expressed in the body frame given current roll/pitch."""
    gx = -math.sin(pitch) * g
    gy =  math.cos(pitch) * math.sin(roll) * g
    gz =  math.cos(pitch) * math.cos(roll) * g
    return gx, gy, gz


# ── Node ──────────────────────────────────────────────────────────────────────

class QuadrupedImuFilter(Node):
    """Quadruped-specialized adaptive IMU filter."""

    def __init__(self):
        super().__init__('quadruped_imu_filter')

        # ── Declare & read parameters ─────────────────────────────────
        self.declare_parameter('accel_alpha',          0.30)
        self.declare_parameter('gyro_alpha',           0.40)
        self.declare_parameter('impact_alpha',         0.08)
        self.declare_parameter('impact_duration_ms',  100.0)
        self.declare_parameter('comp_alpha',           0.98)
        self.declare_parameter('gravity',              9.81)
        self.declare_parameter('outlier_accel',        30.0)
        self.declare_parameter('outlier_gyro',         10.0)
        self.declare_parameter('use_contacts',         True)
        self.declare_parameter('publish_gravity_free', False)

        p = self.get_parameter
        self.accel_alpha   = p('accel_alpha').value
        self.gyro_alpha    = p('gyro_alpha').value
        self.impact_alpha  = p('impact_alpha').value
        self.impact_dur    = p('impact_duration_ms').value * 1e-3   # → seconds
        self.comp_alpha    = p('comp_alpha').value
        self.gravity       = p('gravity').value
        self.out_accel     = p('outlier_accel').value
        self.out_gyro      = p('outlier_gyro').value
        self.use_contacts  = p('use_contacts').value and _HAS_CHAMP_MSGS
        self.pub_gf        = p('publish_gravity_free').value

        # ── State ─────────────────────────────────────────────────────
        self._init        = False
        self._prev_time   = 0.0         # seconds

        # EMA state
        self._ax = self._ay = self._az = 0.0
        self._gx = self._gy = self._gz = 0.0

        # Complementary-filter orientation (Euler, radians)
        self._roll = self._pitch = self._yaw = 0.0

        # Impact tracking
        self._impact_until   = 0.0     # epoch-seconds until impact mode ends
        self._prev_contacts  = [False, False, False, False]

        # ── Publishers ────────────────────────────────────────────────
        self._pub = self.create_publisher(Imu, 'imu/data', 50)
        if self.pub_gf:
            self._pub_gf = self.create_publisher(Imu, 'imu/data/gravity_free', 10)

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(Imu, 'imu/data_raw', self._imu_cb, 50)

        if self.use_contacts:
            self.create_subscription(
                ContactsStamped, 'foot_contacts', self._contact_cb, 10)
            self.get_logger().info(
                'QuadrupedImuFilter: contact-aware mode ENABLED')
        else:
            if not _HAS_CHAMP_MSGS:
                self.get_logger().warn(
                    'QuadrupedImuFilter: champ_msgs not found – '
                    'contact-aware mode disabled.')
            else:
                self.get_logger().info(
                    'QuadrupedImuFilter: contact-aware mode DISABLED by param.')

    # ── Contact callback ──────────────────────────────────────────────────────

    def _contact_cb(self, msg: 'ContactsStamped'):
        """Detect foot touchdowns (rising edges) and enter impact mode."""
        now = self.get_clock().now().nanoseconds * 1e-9
        contacts = list(msg.contacts)

        for i, contact in enumerate(contacts):
            prev = self._prev_contacts[i] if i < len(self._prev_contacts) else False
            if contact and not prev:
                # Rising edge → touchdown detected
                self._impact_until = max(self._impact_until, now + self.impact_dur)

        self._prev_contacts = contacts

    # ── IMU callback ──────────────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu):
        now = self.get_clock().now().nanoseconds * 1e-9
        in_impact = now < self._impact_until

        # ── Initialise on first valid sample ─────────────────────────
        if not self._init:
            self._ax, self._ay, self._az = (
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            )
            self._gx, self._gy, self._gz = (
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            )
            self._roll, self._pitch = _accel_to_roll_pitch(
                self._ax, self._ay, self._az)
            self._yaw = _quat_yaw(msg.orientation)
            self._prev_time = now
            self._init = True
            return

        dt = now - self._prev_time
        self._prev_time = now
        if dt <= 0.0 or dt > 1.0:
            return  # clock jump or stale message

        # ── Outlier / spike rejection ─────────────────────────────────
        da = math.sqrt(
            (msg.linear_acceleration.x - self._ax) ** 2 +
            (msg.linear_acceleration.y - self._ay) ** 2 +
            (msg.linear_acceleration.z - self._az) ** 2)
        dg = math.sqrt(
            (msg.angular_velocity.x - self._gx) ** 2 +
            (msg.angular_velocity.y - self._gy) ** 2 +
            (msg.angular_velocity.z - self._gz) ** 2)

        if da > self.out_accel or dg > self.out_gyro:
            self.get_logger().debug(
                f'Spike rejected – Δa={da:.2f} m/s²  Δω={dg:.2f} rad/s')
            # Publish last good estimate without updating state
            self._publish(msg, in_impact=True)
            return

        # ── Adaptive EMA weights ──────────────────────────────────────
        a = self.impact_alpha if in_impact else self.accel_alpha
        g = self.gyro_alpha  # gyro is less affected by foot impacts

        # ── EMA filtering ─────────────────────────────────────────────
        self._ax = a * msg.linear_acceleration.x + (1.0 - a) * self._ax
        self._ay = a * msg.linear_acceleration.y + (1.0 - a) * self._ay
        self._az = a * msg.linear_acceleration.z + (1.0 - a) * self._az
        self._gx = g * msg.angular_velocity.x    + (1.0 - g) * self._gx
        self._gy = g * msg.angular_velocity.y    + (1.0 - g) * self._gy
        self._gz = g * msg.angular_velocity.z    + (1.0 - g) * self._gz

        # ── Complementary filter for roll / pitch ─────────────────────
        # Step 1 – integrate gyro
        roll_gyro  = self._roll  + self._gx * dt
        pitch_gyro = self._pitch + self._gy * dt
        yaw_gyro   = self._yaw   + self._gz * dt

        # Step 2 – derive tilt from (filtered) accelerometer
        roll_accel, pitch_accel = _accel_to_roll_pitch(
            self._ax, self._ay, self._az)

        # Step 3 – blend (during impact, trust gyro even more)
        ca = min(0.999, self.comp_alpha + 0.01) if in_impact else self.comp_alpha
        self._roll  = ca * roll_gyro  + (1.0 - ca) * roll_accel
        self._pitch = ca * pitch_gyro + (1.0 - ca) * pitch_accel
        self._yaw   = yaw_gyro   # no absolute yaw reference from accel alone

        self._publish(msg, in_impact)

    # ── Publish ───────────────────────────────────────────────────────────────

    def _publish(self, msg: Imu, in_impact: bool):
        ox, oy, oz, ow = _euler_to_quat(self._roll, self._pitch, self._yaw)

        out = Imu()
        out.header = msg.header

        out.linear_acceleration.x = self._ax
        out.linear_acceleration.y = self._ay
        out.linear_acceleration.z = self._az

        out.angular_velocity.x = self._gx
        out.angular_velocity.y = self._gy
        out.angular_velocity.z = self._gz

        out.orientation.x = ox
        out.orientation.y = oy
        out.orientation.z = oz
        out.orientation.w = ow

        # ── Adaptive covariance ───────────────────────────────────────
        # Inflate covariance during impacts → EKF trusts these samples less.
        accel_var  = 1.00  if in_impact else 0.01
        orient_var = 0.10  if in_impact else 0.01
        gyro_var   = 0.001

        out.linear_acceleration_covariance = [
            accel_var, 0.0, 0.0,
            0.0, accel_var, 0.0,
            0.0, 0.0, accel_var,
        ]
        out.angular_velocity_covariance = [
            gyro_var, 0.0, 0.0,
            0.0, gyro_var, 0.0,
            0.0, 0.0, gyro_var,
        ]
        out.orientation_covariance = [
            orient_var,      0.0, 0.0,
            0.0,      orient_var, 0.0,
            0.0,             0.0, orient_var * 10.0,  # yaw has no abs. ref.
        ]

        self._pub.publish(out)

        # ── Optional: gravity-free linear acceleration ────────────────
        if self.pub_gf:
            gx_b, gy_b, gz_b = _gravity_body(self._roll, self._pitch, self.gravity)
            gf = Imu()
            gf.header                        = msg.header
            gf.linear_acceleration.x         = self._ax - gx_b
            gf.linear_acceleration.y         = self._ay - gy_b
            gf.linear_acceleration.z         = self._az - gz_b
            gf.angular_velocity              = out.angular_velocity
            gf.orientation                   = out.orientation
            gf.linear_acceleration_covariance = out.linear_acceleration_covariance
            gf.angular_velocity_covariance    = out.angular_velocity_covariance
            gf.orientation_covariance         = out.orientation_covariance
            self._pub_gf.publish(gf)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = QuadrupedImuFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

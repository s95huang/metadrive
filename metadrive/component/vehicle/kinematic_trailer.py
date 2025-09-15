import math
from typing import Optional

import numpy as np
from panda3d.core import LineSegs, NodePath, Vec3, TransformState, GeomVertexFormat, GeomVertexData, GeomVertexWriter, Geom, GeomTriangles, GeomNode, Material, LColor

from panda3d.bullet import BulletBoxShape

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import Semantics, MetaDriveType, CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.physics_node import BaseRigidBodyNode


class KinematicTrailer(BaseObject):
    """
    Render-only trailer that follows a tractor's hitch using simple kinematics.

    - No physics body or collisions (pure visualization)
    - Pose computed from hitch world position and a persistent trailer axle state
    - Uses provided model if available; otherwise draws a wireframe box
    """

    SEMANTIC_LABEL = Semantics.TRAILER.label
    COLLISION_MASK = CollisionGroup.Vehicle

    def __init__(self, tractor, trailer_config: Optional[dict] = None, name: Optional[str] = None):
        trailer_config = trailer_config or {}
        # Provide a seed; we don't rely on randomness but BaseObject requires a seed
        super().__init__(name=name or f"{tractor.name}_trailer", random_seed=tractor.random_seed, config={})

        # Geometry/config
        self._length = float(trailer_config.get("length", 3.0))
        self._width = float(trailer_config.get("width", 1.6))
        self._height = float(trailer_config.get("height", 1.2))

        # Offsets (in trailer local frame, +X is forward)
        # Vector from trailer origin to hitch (x, y, z)
        self.origin_to_hitch = np.array(trailer_config.get("origin_to_hitch", [self._length / 2.0, 0.0, 0.4]),
                                        dtype=float)

        # Tractor local hitch offset (x, y, z) in tractor local frame
        default_hitch_tractor = [-tractor.LENGTH / 2.0, 0.0, 0.4]
        self.hitch_offset_on_tractor = np.array(trailer_config.get("hitch_offset_on_tractor", default_hitch_tractor),
                                                dtype=float)

        # Optional asset path tuple: (relative_path, scale(x,y,z), offset(x,y,z), hpr(h,p,r))
        # Example: ('trailer/vehicle.gltf', (1,1,1), (0,0,0), (0,0,0))
        self.path = trailer_config.get("path", None)

        # Internal state: previous trailer axle world position for heading update
        # We define axle at the point: origin - (origin_to_hitch projected on X), i.e., along -X from hitch
        self._prev_axle_world: Optional[np.ndarray] = None
        
        # Smoothing and interpolation state for smoother motion
        self._prev_position: Optional[np.ndarray] = None
        self._prev_heading: Optional[float] = None
        self._velocity_history = []  # Track recent velocities for smoothing
        self._position_smoothing_factor = 0.1  # Adjust for smoothness vs responsiveness
        self._heading_smoothing_factor = 0.15  # Different factor for heading
        self._max_velocity_history = 5  # Number of velocity samples to keep

        # Keep tractor reference
        self._tractor = tractor

        # Visualization
        self._add_visualization()

        # Physics body will be created when attached to world to avoid collision at origin

    def _add_visualization(self):
        # Try load model; fallback to wireframe box
        model: Optional[NodePath] = None
        if self.path is not None and isinstance(self.path, (list, tuple)) and len(self.path) == 4:
            try:
                model = self.loader.loadModel(AssetLoader.file_path("models", self.path[0]))
                model.setScale(*self.path[1])
                model.setPos(*self.path[2])
                model.setHpr(*self.path[3])
            except Exception:
                model = None

        if model is None:
            # Build a solid box aligned with local frame: x forward, y left, z up; bottom at z=0
            model = self._build_solid_box(self._length, self._width, self._height)

        # Apply distinctive color for visibility if no PBR trailer asset
        try:
            material = Material()
            base_color = LColor(0.95, 0.55, 0.1, 1.0)  # orange
            material.setBaseColor(base_color)
            material.setDiffuse((base_color[0], base_color[1], base_color[2], 1))
            material.setSpecular((0, 0, 0, 1))
            material.setShininess(2.0)
            model.setMaterial(material, True)
        except Exception:
            pass

        model.reparentTo(self.origin)
        self._node_path_list.append(model)

    def _build_solid_box(self, length: float, width: float, height: float) -> NodePath:
        # Create vertices for a rectangular prism with base at z=0, top at z=height
        fmt = GeomVertexFormat.getV3n3()
        vdata = GeomVertexData('box', fmt, Geom.UHStatic)
        vwriter = GeomVertexWriter(vdata, 'vertex')
        nwriter = GeomVertexWriter(vdata, 'normal')

        hl = length / 2.0
        hw = width / 2.0
        z0, z1 = 0.0, height

        # 8 corners
        corners = [
            Vec3(+hl, +hw, z0), Vec3(+hl, -hw, z0), Vec3(-hl, -hw, z0), Vec3(-hl, +hw, z0),  # bottom
            Vec3(+hl, +hw, z1), Vec3(+hl, -hw, z1), Vec3(-hl, -hw, z1), Vec3(-hl, +hw, z1)   # top
        ]

        # Helper to add a face as two triangles with a normal
        def add_face(idx0, idx1, idx2, idx3, normal):
            start = vwriter.getWriteRow()
            for idx in (idx0, idx1, idx2, idx3):
                vwriter.addData3f(corners[idx])
                nwriter.addData3f(normal)
            tris.addVertices(start + 0, start + 1, start + 2)
            tris.addVertices(start + 2, start + 3, start + 0)

        tris = GeomTriangles(Geom.UHStatic)
        # +X face (right/front)
        add_face(0, 1, 5, 4, Vec3(1, 0, 0))
        # -X face
        add_face(2, 3, 7, 6, Vec3(-1, 0, 0))
        # +Y face (left)
        add_face(3, 0, 4, 7, Vec3(0, 1, 0))
        # -Y face (right)
        add_face(1, 2, 6, 5, Vec3(0, -1, 0))
        # +Z face (top)
        add_face(4, 5, 6, 7, Vec3(0, 0, 1))
        # -Z face (bottom)
        add_face(3, 2, 1, 0, Vec3(0, 0, -1))

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode('trailer_box')
        node.addGeom(geom)
        return NodePath(node)

    def _create_body(self):
        # Create a Bullet rigid body and attach to dynamic world as kinematic
        body = BaseRigidBodyNode(self.name, MetaDriveType.VEHICLE)
        self._node_path_list.append(body)
        shape = BulletBoxShape(Vec3(self._width / 2.0, self._length / 2.0, self._height / 2.0))
        # Place shape so the base touches ground when origin z is ground
        body.addShape(shape, TransformState.makePos(Vec3(0, 0, self._height / 2.0)))
        body.setDeactivationEnabled(False)
        body.setKinematic(True)
        body.setStatic(False)
        body.notifyCollisions(True)
        self.add_body(body)

    def _hitch_world_pos(self) -> np.ndarray:
        # Use BaseVehicle's local-to-world mapping (x forward, y left) for consistency
        hitch_xy = self._tractor.convert_to_world_coordinates(
            self.hitch_offset_on_tractor[:2], self._tractor.position
        )
        hitch_z = self._tractor.get_z() + float(self.hitch_offset_on_tractor[2])
        return np.array([hitch_xy[0], hitch_xy[1], hitch_z], dtype=float)

    def _drawbar_length(self) -> float:
        # Distance from trailer axle to hitch along trailer local +X.
        # Approximate with the length of origin_to_hitch projected on +X if axle at origin.
        return float(np.linalg.norm(self.origin_to_hitch[:2]))

    def update_pose(self):
        """
        Update trailer pose based on the tractor hitch position and previous axle position.
        Handles both forward and reverse motion with smooth interpolation and proper trailer dynamics.
        """
        hitch = self._hitch_world_pos()
        drawbar = self._drawbar_length()
        
        # Track tractor velocity for adaptive smoothing
        tractor_speed = abs(self._tractor.speed) if hasattr(self._tractor, 'speed') else 0
        self._velocity_history.append(tractor_speed)
        if len(self._velocity_history) > self._max_velocity_history:
            self._velocity_history.pop(0)
        
        avg_speed = sum(self._velocity_history) / len(self._velocity_history)
        
        # Adaptive smoothing based on speed - more smoothing at higher speeds
        speed_factor = min(avg_speed / 10.0, 1.0)  # Normalize to 0-1 range
        position_smooth = self._position_smoothing_factor * (1 + speed_factor)
        heading_smooth = self._heading_smoothing_factor * (1 + speed_factor)
        
        # Determine tractor movement direction
        is_forward = self._tractor.is_moving_forward()
        
        # Initialize previous axle directly behind tractor on first call
        if self._prev_axle_world is None:
            heading = self._tractor.heading
            self._prev_axle_world = np.array([hitch[0] - drawbar * heading[0],
                                              hitch[1] - drawbar * heading[1],
                                              hitch[2]], dtype=float)

        # Calculate target axle position based on kinematics
        vec = hitch - self._prev_axle_world
        target_yaw = math.atan2(vec[1], vec[0])
        
        # Different behavior for forward vs reverse
        if is_forward is False:  # Moving backward
            # In reverse, apply more aggressive smoothing to prevent jackknifing
            reverse_smoothing = 0.8
            if hasattr(self, '_last_yaw'):
                yaw_diff = target_yaw - self._last_yaw
                # Handle angle wrap-around
                if yaw_diff > math.pi:
                    yaw_diff -= 2 * math.pi
                elif yaw_diff < -math.pi:
                    yaw_diff += 2 * math.pi
                target_yaw = self._last_yaw + reverse_smoothing * yaw_diff
        
        # Smooth heading transitions
        if self._prev_heading is not None:
            yaw_diff = target_yaw - self._prev_heading
            # Handle angle wrap-around
            if yaw_diff > math.pi:
                yaw_diff -= 2 * math.pi
            elif yaw_diff < -math.pi:
                yaw_diff += 2 * math.pi
            
            # Apply smoothing
            yaw = self._prev_heading + heading_smooth * yaw_diff
        else:
            yaw = target_yaw
        
        self._prev_heading = yaw
        self._last_yaw = yaw
        
        dir_xy = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
        
        # Calculate target axle position
        target_axle = np.array([hitch[0] - drawbar * dir_xy[0],
                               hitch[1] - drawbar * dir_xy[1],
                               hitch[2]], dtype=float)
        
        # Smooth axle position transitions
        if self._prev_axle_world is not None:
            # Interpolate between previous and target axle positions
            alpha = position_smooth
            axle_world = (1 - alpha) * self._prev_axle_world + alpha * target_axle
        else:
            axle_world = target_axle
        
        self._prev_axle_world = axle_world

        # Calculate trailer origin position from axle
        R = np.array([[dir_xy[0], -dir_xy[1]], [dir_xy[1], dir_xy[0]]], dtype=float)
        offset_xy = R @ self.origin_to_hitch[:2]
        target_origin = np.array([hitch[0] - offset_xy[0], hitch[1] - offset_xy[1], hitch[2] - self.origin_to_hitch[2]],
                                dtype=float)
        
        # Smooth origin position
        if self._prev_position is not None:
            # Interpolate between previous and target positions
            alpha = position_smooth
            origin_world = (1 - alpha) * self._prev_position + alpha * target_origin
        else:
            origin_world = target_origin
        
        self._prev_position = origin_world

        # Apply smoothed pose
        self.set_position([origin_world[0], origin_world[1], origin_world[2]])
        self.set_heading_theta(yaw)

    # Convenience alias
    def update(self):
        self.update_pose()

    def attach_to_world(self, parent_node_path, physics_world):
        """Override to create physics body and position correctly when attached"""
        # Create physics body if not already created
        if not hasattr(self, '_bodies') or not self._bodies:
            self._create_body()
        
        # Update position before attaching to avoid spawn collision
        self.update_pose()
        
        # Call parent attach
        super().attach_to_world(parent_node_path, physics_world)

    def destroy(self):
        """Override destroy to ensure proper physics cleanup"""
        # Call parent destroy which handles physics cleanup properly
        super().destroy()

    # Required properties for drawing and utilities
    @property
    def WIDTH(self):
        return self._width

    @property
    def LENGTH(self):
        return self._length

    @property
    def HEIGHT(self):
        return self._height

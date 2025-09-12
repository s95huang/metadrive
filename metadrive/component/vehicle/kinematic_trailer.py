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

        # Keep tractor reference
        self._tractor = tractor

        # Visualization
        self._add_visualization()

        # Physics body (kinematic rigid, box collision)
        self._create_body()

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
        """
        hitch = self._hitch_world_pos()
        drawbar = self._drawbar_length()

        # Initialize previous axle directly behind tractor on first call
        if self._prev_axle_world is None:
            heading = self._tractor.heading
            self._prev_axle_world = np.array([hitch[0] - drawbar * heading[0],
                                              hitch[1] - drawbar * heading[1],
                                              hitch[2]], dtype=float)

        # Compute trailer heading from vector axle->hitch
        vec = hitch - self._prev_axle_world
        yaw = math.atan2(vec[1], vec[0])
        dir_xy = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)

        # New axle position keeps fixed drawbar distance to hitch
        axle_world = np.array([hitch[0] - drawbar * dir_xy[0],
                               hitch[1] - drawbar * dir_xy[1],
                               hitch[2]], dtype=float)
        self._prev_axle_world = axle_world

        # Origin is at hitch minus rotated origin_to_hitch
        R = np.array([[dir_xy[0], -dir_xy[1]], [dir_xy[1], dir_xy[0]]], dtype=float)
        offset_xy = R @ self.origin_to_hitch[:2]
        origin_world = np.array([hitch[0] - offset_xy[0], hitch[1] - offset_xy[1], hitch[2] - self.origin_to_hitch[2]],
                                dtype=float)

        # Apply pose
        self.set_position([origin_world[0], origin_world[1], origin_world[2]])
        self.set_heading_theta(yaw)

    # Convenience alias
    def update(self):
        self.update_pose()

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

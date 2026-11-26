from unittest.mock import patch

try:
    from . import generic as g
except BaseException:
    import generic as g


class PosesTest(g.unittest.TestCase):
    def test_rejected_zero_variance(self):
        mesh = g.trimesh.creation.box()
        for center_mass in ([2, 0, 0], [0.5, 0, 0]):
            # Bound the sampler so a regression fails instead of hanging.
            # Zero covariance always returns the requested mean.
            with patch("trimesh.poses.random_generator") as generator:
                generator.return_value.multivariate_normal.side_effect = [
                    g.np.array([center_mass]),
                    RuntimeError("repeated an impossible zero-variance sample"),
                ]
                with self.assertRaisesRegex(ValueError, "inside the convex hull"):
                    mesh.compute_stable_poses(center_mass=center_mass, sigma=0, seed=0)

    def test_rejected_sample_with_variance(self):
        mesh = g.trimesh.creation.box()
        with patch("trimesh.poses.random_generator") as generator:
            generator.return_value.multivariate_normal.side_effect = [
                g.np.array([[2, 0, 0]]),
                g.np.array([[0, 0, 0]]),
            ]
            transforms, probabilities = mesh.compute_stable_poses(sigma=0.1, seed=0)
        assert transforms.shape == (6, 4, 4)
        g.np.testing.assert_allclose(probabilities, g.np.full(6, 1 / 6))

    def test_nonsampling_poses(self):
        mesh = g.trimesh.creation.icosahedron()

        # Compute the stable poses of the icosahedron
        trans, probs = mesh.compute_stable_poses(seed=0)

        # Probabilities should all be 0.05 (20 faces)
        self.assertTrue(g.np.allclose(g.np.array(probs) - 0.05, 0.0))
        self.assertTrue(len(trans) == 20)
        self.assertTrue(len(probs) == 20)

    def test_multiple(self):
        for mesh in [g.trimesh.creation.icosahedron(), g.get_mesh("unit_cube.STL")]:
            vectors = g.trimesh.util.grid_linspace([[0.0, 0], [1, 1.0]], 5)[1:]
            vectors = g.trimesh.unitize(
                g.np.column_stack((vectors, g.np.ones(len(vectors))))
            )
            for vector, angle in zip(vectors, g.np.linspace(0.0, g.np.pi, len(vectors))):
                matrix = g.trimesh.transformations.rotation_matrix(angle, vector)

                copied = mesh.copy()
                copied.apply_transform(matrix)

                # Compute the stable poses of the icosahedron
                _trans, probs = copied.compute_stable_poses(seed=0)

                # we are only testing primitives with point symmetry
                # AKA 3 principal components of inertia are the same
                facet_count = len(mesh.facets)
                if facet_count == 0:
                    facet_count = len(mesh.faces)
                probability = 1.0 / float(facet_count)

                assert g.np.allclose(g.np.array(probs) - probability, 0.0)

    def test_round(self):
        mesh = g.trimesh.primitives.Cylinder(radius=1.0, height=10.0)

        _transforms, _probabilities = mesh.compute_stable_poses(seed=0)
        _transforms, _probabilities = mesh.compute_stable_poses(n_samples=10, seed=0)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()

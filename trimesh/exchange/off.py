import re

import numpy as np

from ..geometry import triangulate_quads
from ..util import array_to_string, comment_strip, decode_text


def load_off(file_obj, **kwargs) -> dict:
    """
    Load an OFF file into the kwargs for a Trimesh constructor.

    Parameters
    ----------
    file_obj : file object
      Contains an OFF file

    Returns
    ----------
    loaded : dict
      kwargs for Trimesh constructor
    """
    text = file_obj.read()
    # will magically survive weird encoding sometimes
    # comment strip will handle all cases of commenting
    text = comment_strip(decode_text(text)).strip()

    match = re.match(r"(?:COFF|OFF)(?=\s|$)", text)
    if match is None:
        raise ValueError("OFF file is missing the OFF or COFF header")
    raw = text[match.end() :]

    # split into lines and remove whitespace
    splits = [i.strip() for i in str.splitlines(str(raw))]
    # remove empty lines
    splits = [i for i in splits if len(i) > 0]

    # the first non-comment line should be the counts
    if not splits:
        raise ValueError("OFF file is missing the vertex/face count line")
    header = np.array(splits[0].split(), dtype=np.int64)
    if header.size < 2:
        raise ValueError("OFF file has a malformed vertex/face count line")
    vertex_count, face_count = map(int, header[:2])
    if vertex_count < 0 or face_count < 0:
        raise ValueError("OFF vertex and face counts must be nonnegative")
    if len(splits) < 1 + vertex_count + face_count:
        raise ValueError("OFF file ends before its declared vertex or face count")

    vertices = np.array(
        [i.split()[:3] for i in splits[1 : vertex_count + 1]], dtype=np.float64
    )

    # will fail if incorrect number of vertices loaded
    vertices = vertices.reshape((vertex_count, 3))

    # get lines with face data
    faces = [i.split() for i in splits[vertex_count + 1 : vertex_count + face_count + 1]]
    # Values after the declared indices may contain optional face colors.
    for i, line in enumerate(faces):
        count = int(line[0])
        if count < 0 or count > len(line) - 1:
            raise ValueError("OFF face has an invalid vertex count")
        faces[i] = line[1 : count + 1]

    faces = triangulate_quads(faces)
    # faces must reference vertices that actually exist, otherwise the mesh
    # blows up later with a cryptic IndexError and a negative index silently
    # wraps into a wrong face
    if len(faces) and (faces.min() < 0 or faces.max() >= vertex_count):
        raise ValueError("OFF file has a face index outside the range of its vertices")

    # save data as kwargs for a trimesh.Trimesh
    kwargs = {"vertices": vertices, "faces": faces}

    return kwargs


def export_off(mesh, digits=10) -> str:
    """
    Export a mesh as an OFF file, a simple text format

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to export
    digits : int
      Number of digits to include on floats

    Returns
    -----------
    export : str
      OFF format output
    """
    # make sure specified digits is an int
    digits = int(digits)
    # prepend a 3 (face count) to each face
    faces_stacked = np.column_stack((np.ones(len(mesh.faces)) * 3, mesh.faces)).astype(
        np.int64
    )
    # the header is vertex count, face count, another number
    export = "\n".join(
        [
            "OFF",
            str(len(mesh.vertices)) + " " + str(len(mesh.faces)) + " 0",
            array_to_string(mesh.vertices, col_delim=" ", row_delim="\n", digits=digits),
            array_to_string(faces_stacked, col_delim=" ", row_delim="\n"),
            "",
        ]
    )

    return export


_off_loaders = {"off": load_off}
_off_exporters = {"off": export_off}

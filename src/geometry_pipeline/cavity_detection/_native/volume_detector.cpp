#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/intersections.h>

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_2.h>
#include <CGAL/Constrained_triangulation_face_base_2.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/Triangulation_data_structure_2.h>

#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/Eigen_diagonalize_traits.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Kernel   = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3  = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;
using Plane_3  = Kernel::Plane_3;
using Triangle_3 = Kernel::Triangle_3;
using Segment_3  = Kernel::Segment_3;
using Bbox_3     = CGAL::Bbox_3;

using Primitive   = CGAL::AABB_triangle_primitive_3<Kernel, std::vector<Triangle_3>::const_iterator>;
using AABB_traits = CGAL::AABB_traits_3<Kernel, Primitive>;
using Tree        = CGAL::AABB_tree<AABB_traits>;

using K2 = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_2 = K2::Point_2;

struct ObjFace
{
    std::size_t original_face_id = 0;
    std::vector<std::size_t> vertex_indices;
};

struct TriMeta
{
    std::size_t original_face_id = 0;
};

struct OriginalFaceGeometry
{
    std::size_t original_face_id = 0;
    Vector_3 normal;
    Point_3 preferred_probe_point;              // barycenter of largest interior triangle
    std::vector<Point_3> interior_probe_points; // barycenters of interior triangles
};

struct VolumeMappingResult
{
    std::size_t bounded_volume_count = 0;
    std::vector<std::vector<std::size_t>> volume_to_original_faces;
    // Parallel to volume_to_original_faces: Gmsh-style orientation sign for
    // each (volume, face). +1 if the face normal points OUT of the volume,
    // -1 if it points INTO the volume.
    std::vector<std::vector<int>> volume_to_face_signs;
    std::vector<std::vector<std::size_t>> original_face_to_volumes;
    std::vector<char> volume_is_manifold; // 1 = true, 0 = false
};

struct Grid
{
    double min_x = 0.0;
    double min_y = 0.0;
    double min_z = 0.0;
    double step  = 1.0;
    int nx = 0;
    int ny = 0;
    int nz = 0;

    std::vector<char> free_mask;
    std::vector<int> component_id;
    std::vector<char> is_exterior_comp;
};

struct FaceInfo2
{
    bool in_domain = false;
};

using Vb2  = CGAL::Triangulation_vertex_base_2<K2>;
using Fbb2 = CGAL::Triangulation_face_base_2<K2>;
using Fb2  = CGAL::Constrained_triangulation_face_base_2<K2, Fbb2>;
using Fbi2 = CGAL::Triangulation_face_base_with_info_2<FaceInfo2, K2, Fb2>;
using Tds2 = CGAL::Triangulation_data_structure_2<Vb2, Fbi2>;
using CDT2 = CGAL::Constrained_Delaunay_triangulation_2<K2, Tds2>;

static void require(bool cond, const std::string& msg)
{
    if (!cond)
    {
        throw std::runtime_error(msg);
    }
}

static std::vector<std::string> split_ws(const std::string& line)
{
    std::istringstream iss(line);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok)
    {
        out.push_back(tok);
    }
    return out;
}

static std::size_t parse_obj_index_token(const std::string& tok, std::size_t vertex_count_so_far)
{
    std::string first = tok;
    const std::size_t slash = tok.find('/');
    if (slash != std::string::npos)
    {
        first = tok.substr(0, slash);
    }

    require(!first.empty(), "Invalid OBJ face token: empty vertex index.");

    long long idx = std::stoll(first);
    long long resolved = 0;

    if (idx > 0)
    {
        resolved = idx - 1;
    }
    else
    {
        resolved = static_cast<long long>(vertex_count_so_far) + idx;
    }

    require(resolved >= 0, "OBJ face index resolved to negative.");
    return static_cast<std::size_t>(resolved);
}

static void read_obj(
    const std::string& path,
    std::vector<Point_3>& vertices,
    std::vector<ObjFace>& faces)
{
    std::ifstream in(path);
    require(in.good(), "Could not open OBJ file: " + path);

    std::string line;
    std::size_t face_id = 0;

    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        const auto toks = split_ws(line);
        if (toks.empty()) continue;

        if (toks[0] == "v")
        {
            require(toks.size() >= 4, "Invalid vertex line in OBJ.");
            vertices.emplace_back(std::stod(toks[1]), std::stod(toks[2]), std::stod(toks[3]));
        }
        else if (toks[0] == "f")
        {
            require(toks.size() >= 4, "OBJ face must have at least 3 vertices.");

            ObjFace f;
            f.original_face_id = face_id++;

            for (std::size_t i = 1; i < toks.size(); ++i)
            {
                const std::size_t vi = parse_obj_index_token(toks[i], vertices.size());
                require(vi < vertices.size(), "OBJ face index out of range.");
                f.vertex_indices.push_back(vi);
            }

            faces.push_back(std::move(f));
        }
    }

    require(!vertices.empty(), "OBJ has no vertices.");
    require(!faces.empty(), "OBJ has no faces.");
}

// ------------------------------
// Mesh-IR JSON input
// ------------------------------
//
// The Python side serializes its internal mesh representation directly to a
// small JSON document instead of going through OBJ text:
//
//   {
//     "vertices": [[x, y, z], ...],
//     "faces":    [[i0, i1, i2, ...], ...]   // 0-based indices into vertices
//   }
//
// This is a minimal, strict reader scoped to exactly that schema (we control
// the writer). It is intentionally not a general-purpose JSON library.

struct JsonReader
{
    const std::string& s;
    std::size_t i = 0;

    explicit JsonReader(const std::string& str) : s(str) {}

    void skip_ws()
    {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        {
            ++i;
        }
    }

    bool match(char c)
    {
        skip_ws();
        if (i < s.size() && s[i] == c)
        {
            ++i;
            return true;
        }
        return false;
    }

    void expect(char c)
    {
        require(match(c), std::string("JSON mesh: expected '") + c + "'.");
    }

    std::string parse_string()
    {
        skip_ws();
        require(i < s.size() && s[i] == '"', "JSON mesh: expected string.");
        ++i;
        std::string out;
        while (i < s.size() && s[i] != '"')
        {
            char c = s[i++];
            if (c == '\\' && i < s.size())
            {
                char e = s[i++];
                switch (e)
                {
                    case 'n':  out += '\n'; break;
                    case 't':  out += '\t'; break;
                    case 'r':  out += '\r'; break;
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    default:   out += e;    break;
                }
            }
            else
            {
                out += c;
            }
        }
        require(i < s.size(), "JSON mesh: unterminated string.");
        ++i;
        return out;
    }

    double parse_number()
    {
        skip_ws();
        const std::size_t start = i;
        while (i < s.size())
        {
            const char c = s[i];
            if (std::isdigit(static_cast<unsigned char>(c)) ||
                c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
            {
                ++i;
            }
            else
            {
                break;
            }
        }
        require(i > start, "JSON mesh: expected number.");
        return std::stod(s.substr(start, i - start));
    }
};

static void read_mesh_json(
    const std::string& path,
    std::vector<Point_3>& vertices,
    std::vector<ObjFace>& faces)
{
    std::ifstream in(path);
    require(in.good(), "Could not open JSON mesh file: " + path);

    std::stringstream buf;
    buf << in.rdbuf();
    const std::string content = buf.str();

    JsonReader r(content);
    r.expect('{');

    bool got_vertices = false;
    bool got_faces = false;

    while (true)
    {
        const std::string key = r.parse_string();
        r.expect(':');

        if (key == "vertices")
        {
            r.expect('[');
            if (!r.match(']'))
            {
                do
                {
                    r.expect('[');
                    const double x = r.parse_number();
                    r.expect(',');
                    const double y = r.parse_number();
                    r.expect(',');
                    const double z = r.parse_number();
                    r.expect(']');
                    vertices.emplace_back(x, y, z);
                } while (r.match(','));
                r.expect(']');
            }
            got_vertices = true;
        }
        else if (key == "faces")
        {
            r.expect('[');
            std::size_t face_id = 0;
            if (!r.match(']'))
            {
                do
                {
                    r.expect('[');
                    ObjFace f;
                    f.original_face_id = face_id++;

                    const double first = r.parse_number();
                    f.vertex_indices.push_back(
                        static_cast<std::size_t>(std::llround(first)));
                    while (r.match(','))
                    {
                        const double k = r.parse_number();
                        f.vertex_indices.push_back(
                            static_cast<std::size_t>(std::llround(k)));
                    }
                    r.expect(']');
                    faces.push_back(std::move(f));
                } while (r.match(','));
                r.expect(']');
            }
            got_faces = true;
        }
        else
        {
            require(false, "JSON mesh: unsupported key '" + key + "'.");
        }

        if (r.match(','))
        {
            continue;
        }
        break;
    }

    r.expect('}');

    require(got_vertices && got_faces,
            "JSON mesh must contain both 'vertices' and 'faces'.");
    require(!vertices.empty(), "JSON mesh has no vertices.");
    require(!faces.empty(), "JSON mesh has no faces.");

    for (const ObjFace& f : faces)
    {
        require(f.vertex_indices.size() >= 3,
                "JSON mesh face has fewer than 3 vertices.");
        for (std::size_t vi : f.vertex_indices)
        {
            require(vi < vertices.size(), "JSON mesh face index out of range.");
        }
    }
}

static bool path_has_suffix_ci(const std::string& path, const std::string& suffix)
{
    if (path.size() < suffix.size())
    {
        return false;
    }
    return std::equal(
        suffix.rbegin(), suffix.rend(), path.rbegin(),
        [](char a, char b)
        {
            return std::tolower(static_cast<unsigned char>(a)) ==
                   std::tolower(static_cast<unsigned char>(b));
        });
}

// Dispatch by file extension: ".json" -> mesh-IR JSON, anything else -> OBJ.
static void read_mesh(
    const std::string& path,
    std::vector<Point_3>& vertices,
    std::vector<ObjFace>& faces)
{
    if (path_has_suffix_ci(path, ".json"))
    {
        read_mesh_json(path, vertices, faces);
    }
    else
    {
        read_obj(path, vertices, faces);
    }
}

static Vector_3 compute_polygon_normal(
    const ObjFace& face,
    const std::vector<Point_3>& vertices)
{
    double nx = 0.0, ny = 0.0, nz = 0.0;
    const std::size_t n = face.vertex_indices.size();

    for (std::size_t i = 0; i < n; ++i)
    {
        const Point_3& a = vertices[face.vertex_indices[i]];
        const Point_3& b = vertices[face.vertex_indices[(i + 1) % n]];

        nx += (a.y() - b.y()) * (a.z() + b.z());
        ny += (a.z() - b.z()) * (a.x() + b.x());
        nz += (a.x() - b.x()) * (a.y() + b.y());
    }

    Vector_3 nrm(nx, ny, nz);
    const double len2 = nrm.squared_length();
    require(len2 > 0.0, "Degenerate face with zero normal.");
    return nrm / std::sqrt(len2);
}

static std::vector<std::size_t> remove_consecutive_duplicate_indices(
    const std::vector<std::size_t>& indices,
    const std::vector<Point_3>& vertices)
{
    std::vector<std::size_t> out;
    if (indices.empty()) return out;

    auto same_point = [&](std::size_t a, std::size_t b) -> bool
    {
        return vertices[a] == vertices[b];
    };

    for (std::size_t idx : indices)
    {
        if (out.empty() || !same_point(out.back(), idx))
        {
            out.push_back(idx);
        }
    }

    if (out.size() >= 2 && same_point(out.front(), out.back()))
    {
        out.pop_back();
    }

    return out;
}
struct UndirectedEdge
{
    std::size_t a = 0;
    std::size_t b = 0;

    UndirectedEdge() = default;

    UndirectedEdge(std::size_t u, std::size_t v)
    {
        if (u < v)
        {
            a = u;
            b = v;
        }
        else
        {
            a = v;
            b = u;
        }
    }

    bool operator<(const UndirectedEdge& other) const
    {
        if (a != other.a) return a < other.a;
        return b < other.b;
    }
};

static std::vector<char> detect_volume_manifoldness(
    const VolumeMappingResult& result,
    const std::vector<ObjFace>& faces)
{
    std::vector<char> out(result.bounded_volume_count, 1);

    for (std::size_t vid = 0; vid < result.bounded_volume_count; ++vid)
    {
        std::map<UndirectedEdge, int> edge_count;

        const std::vector<std::size_t>& face_ids = result.volume_to_original_faces[vid];

        for (std::size_t fid : face_ids)
        {
            require(fid < faces.size(),
                    "Face id out of range in detect_volume_manifoldness().");

            const ObjFace& face = faces[fid];
            require(face.vertex_indices.size() >= 3,
                    "Face has fewer than 3 vertices in detect_volume_manifoldness().");

            for (std::size_t i = 0; i < face.vertex_indices.size(); ++i)
            {
                const std::size_t va = face.vertex_indices[i];
                const std::size_t vb = face.vertex_indices[(i + 1) % face.vertex_indices.size()];

                const UndirectedEdge e(va, vb);
                edge_count[e] += 1;
            }
        }

        bool is_manifold = true;
        for (const auto& kv : edge_count)
        {
            if (kv.second != 2)
            {
                is_manifold = false;
                break;
            }
        }

        out[vid] = static_cast<char>(is_manifold ? 1 : 0);
    }

    return out;
}
template <class CDT>
static void mark_domain(CDT& cdt)
{
    for (auto f = cdt.all_faces_begin(); f != cdt.all_faces_end(); ++f)
    {
        f->info().in_domain = false;
    }

    std::map<typename CDT::Face_handle, int> nesting_level;
    std::list<typename CDT::Edge> border;
    std::queue<typename CDT::Face_handle> q;

    typename CDT::Face_handle infinite = cdt.infinite_face();
    nesting_level[infinite] = 0;
    q.push(infinite);

    while (!q.empty())
    {
        typename CDT::Face_handle fh = q.front();
        q.pop();

        for (int i = 0; i < 3; ++i)
        {
            typename CDT::Face_handle n = fh->neighbor(i);
            if (nesting_level.find(n) != nesting_level.end())
            {
                continue;
            }

            if (!cdt.is_constrained(std::make_pair(fh, i)))
            {
                nesting_level[n] = nesting_level[fh];
                q.push(n);
            }
            else
            {
                border.push_back(std::make_pair(fh, i));
            }
        }
    }

    while (!border.empty())
    {
        typename CDT::Edge e = border.front();
        border.pop_front();

        typename CDT::Face_handle n = e.first->neighbor(e.second);
        if (nesting_level.find(n) != nesting_level.end())
        {
            continue;
        }

        nesting_level[n] = nesting_level[e.first] + 1;
        q.push(n);

        while (!q.empty())
        {
            typename CDT::Face_handle fh = q.front();
            q.pop();

            for (int i = 0; i < 3; ++i)
            {
                typename CDT::Face_handle nn = fh->neighbor(i);
                if (nesting_level.find(nn) != nesting_level.end())
                {
                    continue;
                }

                if (!cdt.is_constrained(std::make_pair(fh, i)))
                {
                    nesting_level[nn] = nesting_level[fh];
                    q.push(nn);
                }
                else
                {
                    border.push_back(std::make_pair(fh, i));
                }
            }
        }
    }

    for (auto f = cdt.finite_faces_begin(); f != cdt.finite_faces_end(); ++f)
    {
        auto it = nesting_level.find(f);
        if (it != nesting_level.end())
        {
            f->info().in_domain = ((it->second % 2) == 1);
        }
    }
}

static double norm3(const Vector_3& v)
{
    return std::sqrt(CGAL::to_double(v.squared_length()));
}

static Vector_3 normalized3(const Vector_3& v)
{
    const double n = norm3(v);
    require(n > 0.0, "Cannot normalize zero vector.");
    return v / n;
}

static Vector_3 cross3(const Vector_3& a, const Vector_3& b)
{
    return CGAL::cross_product(a, b);
}

static double dot3(const Vector_3& a, const Vector_3& b)
{
    return CGAL::to_double(a * b);
}

static double squared_distance_points(const Point_3& a, const Point_3& b)
{
    return CGAL::to_double(CGAL::squared_distance(a, b));
}

static double triangle_area(const Point_3& a, const Point_3& b, const Point_3& c)
{
    const Vector_3 ab = b - a;
    const Vector_3 ac = c - a;
    return 0.5 * norm3(cross3(ab, ac));
}

static Point_3 triangle_barycenter(const Point_3& a, const Point_3& b, const Point_3& c)
{
    return Point_3(
        (a.x() + b.x() + c.x()) / 3.0,
        (a.y() + b.y() + c.y()) / 3.0,
        (a.z() + b.z() + c.z()) / 3.0
    );
}

struct FacePlaneFrame
{
    Point_3 origin;
    Vector_3 u;
    Vector_3 v;
    Vector_3 n;
};
static FacePlaneFrame build_best_fit_plane_frame(
    const ObjFace& face,
    const std::vector<Point_3>& vertices)
{
    std::vector<Point_3> pts;
    pts.reserve(face.vertex_indices.size());
    for (std::size_t vi : face.vertex_indices)
    {
        pts.push_back(vertices[vi]);
    }

    Plane_3 plane;
    Point_3 centroid;
    CGAL::linear_least_squares_fitting_3(
        pts.begin(),
        pts.end(),
        plane,
        centroid,
        CGAL::Dimension_tag<0>(),
        Kernel(),
        CGAL::Eigen_diagonalize_traits<double>()
    );

    Vector_3 n = plane.orthogonal_vector();
    require(n.squared_length() > 0.0,
            "Best-fit plane normal is zero for face " + std::to_string(face.original_face_id) + ".");

    n = normalized3(n);

    Vector_3 hint = compute_polygon_normal(face, vertices);
    if (dot3(n, hint) < 0.0)
    {
        n = -n;
    }

    Vector_3 ref;
    if (std::abs(n.x()) < 0.9)
    {
        ref = Vector_3(1.0, 0.0, 0.0);
    }
    else
    {
        ref = Vector_3(0.0, 1.0, 0.0);
    }

    Vector_3 u = cross3(n, ref);
    if (u.squared_length() == 0.0)
    {
        ref = Vector_3(0.0, 0.0, 1.0);
        u = cross3(n, ref);
    }
    u = normalized3(u);

    Vector_3 v = cross3(n, u);
    v = normalized3(v);

    return {centroid, u, v, n};
}

static Point_2 project_to_2d(const Point_3& p, const FacePlaneFrame& frame)
{
    const Vector_3 d = p - frame.origin;
    return Point_2(dot3(d, frame.u), dot3(d, frame.v));
}

static Point_3 lift_to_3d(const Point_2& p, const FacePlaneFrame& frame)
{
    return frame.origin + p.x() * frame.u + p.y() * frame.v;
}

static bool polygon_has_duplicate_2d_points(const std::vector<Point_2>& pts, double eps = 1e-12)
{
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        for (std::size_t j = i + 1; j < pts.size(); ++j)
        {
            const double dx = pts[i].x() - pts[j].x();
            const double dy = pts[i].y() - pts[j].y();
            if ((dx * dx + dy * dy) <= eps * eps)
            {
                return true;
            }
        }
    }
    return false;
}

static void deduplicate_points(std::vector<Point_3>& pts, double eps = 1e-12)
{
    std::vector<Point_3> out;
    out.reserve(pts.size());

    for (const Point_3& p : pts)
    {
        bool duplicate = false;
        for (const Point_3& q : out)
        {
            if (squared_distance_points(p, q) <= eps * eps)
            {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
        {
            out.push_back(p);
        }
    }

    pts.swap(out);
}

static void triangulate_one_face_with_best_fit_cdt(
    const ObjFace& face,
    const std::vector<Point_3>& vertices,
    std::vector<Triangle_3>& triangles,
    std::vector<TriMeta>& tri_meta,
    std::vector<OriginalFaceGeometry>& face_geom)
{
    std::vector<std::size_t> cleaned =
        remove_consecutive_duplicate_indices(face.vertex_indices, vertices);

    require(cleaned.size() >= 3,
            "Face " + std::to_string(face.original_face_id) +
            " has fewer than 3 distinct vertices.");

    ObjFace clean_face = face;
    clean_face.vertex_indices = cleaned;

    const Vector_3 normal = compute_polygon_normal(clean_face, vertices);
    const FacePlaneFrame frame = build_best_fit_plane_frame(clean_face, vertices);

    std::vector<Point_2> poly2;
    poly2.reserve(cleaned.size());
    for (std::size_t vi : cleaned)
    {
        poly2.push_back(project_to_2d(vertices[vi], frame));
    }

    require(!polygon_has_duplicate_2d_points(poly2),
            "Face " + std::to_string(face.original_face_id) +
            " collapses to duplicate points in best-fit 2D projection.");

    CDT2 cdt;
    std::vector<CDT2::Vertex_handle> vhs;
    vhs.reserve(poly2.size());

    for (const Point_2& p : poly2)
    {
        vhs.push_back(cdt.insert(p));
    }

    for (std::size_t i = 0; i < vhs.size(); ++i)
    {
        cdt.insert_constraint(vhs[i], vhs[(i + 1) % vhs.size()]);
    }

    mark_domain(cdt);

    std::size_t inside_count = 0;
    double best_area = -1.0;
    Point_3 best_probe_point;
    std::vector<Point_3> probe_points;

    for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit)
    {
        if (!fit->info().in_domain)
        {
            continue;
        }

        const Point_2 q0 = fit->vertex(0)->point();
        const Point_2 q1 = fit->vertex(1)->point();
        const Point_2 q2 = fit->vertex(2)->point();

        const Point_3 p0 = lift_to_3d(q0, frame);
        const Point_3 p1 = lift_to_3d(q1, frame);
        const Point_3 p2 = lift_to_3d(q2, frame);

        Triangle_3 tri(p0, p1, p2);
        if (tri.is_degenerate())
        {
            continue;
        }

        triangles.push_back(tri);
        tri_meta.push_back({face.original_face_id});
        ++inside_count;

        const double area = triangle_area(p0, p1, p2);
        const Point_3 bary = triangle_barycenter(p0, p1, p2);
        probe_points.push_back(bary);

        if (area > best_area)
        {
            best_area = area;
            best_probe_point = bary;
        }
    }

    require(inside_count > 0,
            "Best-fit CDT produced no interior triangles for face " +
            std::to_string(face.original_face_id) + ".");

    deduplicate_points(probe_points);

    require(!probe_points.empty(),
            "No valid interior probe points for face " + std::to_string(face.original_face_id) + ".");

    // Put the preferred point first.
    std::vector<Point_3> ordered_probe_points;
    ordered_probe_points.reserve(probe_points.size());
    ordered_probe_points.push_back(best_probe_point);
    for (const Point_3& p : probe_points)
    {
        if (squared_distance_points(p, best_probe_point) > 1e-24)
        {
            ordered_probe_points.push_back(p);
        }
    }

    face_geom.push_back({
        face.original_face_id,
        normal,
        best_probe_point,
        ordered_probe_points
    });
}

static void triangulate_faces(
    const std::vector<Point_3>& vertices,
    const std::vector<ObjFace>& faces,
    std::vector<Triangle_3>& triangles,
    std::vector<TriMeta>& tri_meta,
    std::vector<OriginalFaceGeometry>& face_geom)
{
    triangles.clear();
    tri_meta.clear();
    face_geom.clear();
    face_geom.reserve(faces.size());

    for (const ObjFace& face : faces)
    {
        require(face.vertex_indices.size() >= 3,
                "Face " + std::to_string(face.original_face_id) + " has fewer than 3 vertices.");

        triangulate_one_face_with_best_fit_cdt(face, vertices, triangles, tri_meta, face_geom);
    }

    require(face_geom.size() == faces.size(), "face_geom size mismatch after triangulation.");
}

static Bbox_3 compute_bbox(const std::vector<Point_3>& vertices)
{
    Bbox_3 box = vertices.front().bbox();
    for (std::size_t i = 1; i < vertices.size(); ++i)
    {
        box = box + vertices[i].bbox();
    }
    return box;
}

static int grid_index(const Grid& g, int ix, int iy, int iz)
{
    return (iz * g.ny + iy) * g.nx + ix;
}

static Point_3 grid_point(const Grid& g, int ix, int iy, int iz)
{
    return Point_3(
        g.min_x + (static_cast<double>(ix) + 0.5) * g.step,
        g.min_y + (static_cast<double>(iy) + 0.5) * g.step,
        g.min_z + (static_cast<double>(iz) + 0.5) * g.step
    );
}

static Grid build_free_space_grid(
    const Bbox_3& bbox,
    const Tree& tree,
    int target_resolution,
    double clearance_factor)
{
    require(target_resolution >= 10, "Grid resolution too small. Use at least 10.");
    require(clearance_factor > 0.0, "clearance_factor must be > 0.");

    const double dx = bbox.xmax() - bbox.xmin();
    const double dy = bbox.ymax() - bbox.ymin();
    const double dz = bbox.zmax() - bbox.zmin();
    const double max_dim = std::max({dx, dy, dz});
    require(max_dim > 0.0, "Degenerate bounding box.");

    Grid g;
    g.step = max_dim / static_cast<double>(target_resolution);

    const double pad = 2.0 * g.step;
    g.min_x = bbox.xmin() - pad;
    g.min_y = bbox.ymin() - pad;
    g.min_z = bbox.zmin() - pad;

    const double max_x = bbox.xmax() + pad;
    const double max_y = bbox.ymax() + pad;
    const double max_z = bbox.zmax() + pad;

    g.nx = static_cast<int>(std::ceil((max_x - g.min_x) / g.step));
    g.ny = static_cast<int>(std::ceil((max_y - g.min_y) / g.step));
    g.nz = static_cast<int>(std::ceil((max_z - g.min_z) / g.step));

    require(g.nx > 2 && g.ny > 2 && g.nz > 2, "Grid too small after setup.");

    const int total = g.nx * g.ny * g.nz;
    g.free_mask.assign(total, 0);
    g.component_id.assign(total, -1);

    const double clearance = clearance_factor * g.step;
    const double clearance2 = clearance * clearance;

    for (int iz = 0; iz < g.nz; ++iz)
    {
        for (int iy = 0; iy < g.ny; ++iy)
        {
            for (int ix = 0; ix < g.nx; ++ix)
            {
                const Point_3 p = grid_point(g, ix, iy, iz);
                const double d2 = tree.squared_distance(p);

                if (d2 > clearance2)
                {
                    g.free_mask[grid_index(g, ix, iy, iz)] = 1;
                }
            }
        }
    }

    return g;
}

static void label_free_space_components(Grid& g, const Tree& tree)
{
    const std::array<std::array<int, 3>, 6> dirs = {{
        {{ 1, 0, 0}}, {{-1, 0, 0}},
        {{ 0, 1, 0}}, {{ 0,-1, 0}},
        {{ 0, 0, 1}}, {{ 0, 0,-1}}
    }};

    int next_component = 0;
    g.is_exterior_comp.clear();

    for (int iz = 0; iz < g.nz; ++iz)
    {
        for (int iy = 0; iy < g.ny; ++iy)
        {
            for (int ix = 0; ix < g.nx; ++ix)
            {
                const int start_idx = grid_index(g, ix, iy, iz);
                if (!g.free_mask[start_idx]) continue;
                if (g.component_id[start_idx] != -1) continue;

                std::queue<std::array<int, 3>> q;
                q.push({ix, iy, iz});
                g.component_id[start_idx] = next_component;

                bool touches_boundary = false;

                while (!q.empty())
                {
                    const auto cur = q.front();
                    q.pop();

                    const int cx = cur[0];
                    const int cy = cur[1];
                    const int cz = cur[2];

                    if (cx == 0 || cy == 0 || cz == 0 ||
                        cx == g.nx - 1 || cy == g.ny - 1 || cz == g.nz - 1)
                    {
                        touches_boundary = true;
                    }

                    const Point_3 p = grid_point(g, cx, cy, cz);

                    for (const auto& d : dirs)
                    {
                        const int nx = cx + d[0];
                        const int ny = cy + d[1];
                        const int nz = cz + d[2];

                        if (nx < 0 || ny < 0 || nz < 0 ||
                            nx >= g.nx || ny >= g.ny || nz >= g.nz)
                        {
                            continue;
                        }

                        const int nidx = grid_index(g, nx, ny, nz);
                        if (!g.free_mask[nidx]) continue;
                        if (g.component_id[nidx] != -1) continue;

                        const Point_3 qpt = grid_point(g, nx, ny, nz);
                        const Segment_3 seg(p, qpt);

                        if (tree.do_intersect(seg))
                        {
                            continue;
                        }

                        g.component_id[nidx] = next_component;
                        q.push({nx, ny, nz});
                    }
                }

                g.is_exterior_comp.push_back(static_cast<char>(touches_boundary ? 1 : 0));
                ++next_component;
            }
        }
    }
}

static std::array<int, 3> nearest_grid_coords(const Grid& g, const Point_3& p)
{
    auto to_idx = [&](double c, double minc, int n) -> int
    {
        const double t = (c - minc) / g.step - 0.5;
        int idx = static_cast<int>(std::llround(t));
        idx = std::max(0, std::min(n - 1, idx));
        return idx;
    };

    return {
        to_idx(p.x(), g.min_x, g.nx),
        to_idx(p.y(), g.min_y, g.ny),
        to_idx(p.z(), g.min_z, g.nz)
    };
}

static int find_nearby_component_visible(
    const Grid& g,
    const Tree& tree,
    const Point_3& p,
    int search_radius_cells)
{
    const auto seed = nearest_grid_coords(g, p);

    int best_comp = -1;
    double best_dist2 = std::numeric_limits<double>::infinity();

    for (int r = 0; r <= search_radius_cells; ++r)
    {
        bool found_this_radius = false;

        for (int dz = -r; dz <= r; ++dz)
        {
            for (int dy = -r; dy <= r; ++dy)
            {
                for (int dx = -r; dx <= r; ++dx)
                {
                    const int ix = seed[0] + dx;
                    const int iy = seed[1] + dy;
                    const int iz = seed[2] + dz;

                    if (ix < 0 || iy < 0 || iz < 0 ||
                        ix >= g.nx || iy >= g.ny || iz >= g.nz)
                    {
                        continue;
                    }

                    const int idx = grid_index(g, ix, iy, iz);
                    const int comp = g.component_id[idx];
                    if (comp < 0) continue;

                    const Point_3 gp = grid_point(g, ix, iy, iz);
                    const Segment_3 seg(p, gp);

                    if (tree.do_intersect(seg))
                    {
                        continue;
                    }

                    const double d2 = CGAL::to_double(CGAL::squared_distance(p, gp));
                    if (d2 < best_dist2)
                    {
                        best_dist2 = d2;
                        best_comp = comp;
                        found_this_radius = true;
                    }
                }
            }
        }

        if (found_this_radius)
        {
            return best_comp;
        }
    }

    return best_comp;
}

static VolumeMappingResult map_faces_to_bounded_volumes(
    const std::vector<OriginalFaceGeometry>& face_geom,
    const Grid& g,
    const Tree& tree,
    const std::vector<double>& face_probe_distance_factors,
    int component_search_radius_cells)
{
    require(!face_probe_distance_factors.empty(), "face_probe_distance_factors must not be empty.");
    for (double f : face_probe_distance_factors)
    {
        require(f > 0.0, "All face_probe_distance_factors must be > 0.");
    }

    std::vector<int> comp_to_volume(g.is_exterior_comp.size(), -1);
    int next_volume_id = 0;
    for (std::size_t comp = 0; comp < g.is_exterior_comp.size(); ++comp)
    {
        if (!g.is_exterior_comp[comp])
        {
            comp_to_volume[comp] = next_volume_id++;
        }
    }

    VolumeMappingResult out;
    out.bounded_volume_count = static_cast<std::size_t>(next_volume_id);
    out.volume_to_original_faces.resize(out.bounded_volume_count);
    out.volume_to_face_signs.resize(out.bounded_volume_count);
    out.original_face_to_volumes.resize(face_geom.size());

    for (const auto& fg : face_geom)
    {
        // volume id -> orientation sign for this face.
        // -1 : face normal points INTO the volume (probe along +normal hit it)
        // +1 : face normal points OUT of the volume (probe along -normal hit it)
        std::map<std::size_t, int> vol_sign;

        for (const Point_3& base : fg.interior_probe_points)
        {
            for (double dist_factor : face_probe_distance_factors)
            {
                const double probe_dist = dist_factor * g.step;

                const Point_3 p_plus  = base + probe_dist * fg.normal;
                const Point_3 p_minus = base - probe_dist * fg.normal;

                const int comp_plus =
                    find_nearby_component_visible(g, tree, p_plus, component_search_radius_cells);
                const int comp_minus =
                    find_nearby_component_visible(g, tree, p_minus, component_search_radius_cells);

                if (comp_plus >= 0 && !g.is_exterior_comp[comp_plus])
                {
                    const std::size_t vid = static_cast<std::size_t>(comp_to_volume[comp_plus]);
                    vol_sign.emplace(vid, -1); // keep first sign seen
                }
                if (comp_minus >= 0 && !g.is_exterior_comp[comp_minus])
                {
                    const std::size_t vid = static_cast<std::size_t>(comp_to_volume[comp_minus]);
                    vol_sign.emplace(vid, +1); // keep first sign seen
                }
            }
        }

        // std::map iterates in ascending key order, so the per-face volume
        // list is naturally sorted and de-duplicated.
        auto& dst = out.original_face_to_volumes[fg.original_face_id];
        dst.reserve(vol_sign.size());
        for (const auto& kv : vol_sign)
        {
            dst.push_back(kv.first);
            out.volume_to_original_faces[kv.first].push_back(fg.original_face_id);
            out.volume_to_face_signs[kv.first].push_back(kv.second);
        }
    }

    return out;
}

// ------------------------------
// Multi-scale (room + furniture) detection
// ------------------------------
//
// A single global uniform grid cannot resolve a large room AND a small cavity
// at the same time: the cell size is `max_dim / target_resolution`, so a tiny
// enclosed region whose half-width is below `clearance_factor * step` yields
// zero free cells and is never discovered, no matter how the probes are tuned.
//
// Fix: drive detection from MESH CONNECTIVITY, not from coarse free space.
//   1. Split the mesh into connected surface shells via shared-edge adjacency.
//      Furniture that is separate geometry from the room walls becomes its own
//      shell automatically.
//   2. For each shell, run the existing grid + visibility detector on a LOCAL
//      grid whose cell size is relative to *that shell's* bounding box. Small
//      shells therefore get a fine grid, the big room gets a coarse one, and
//      each shell's cost stays bounded by its own `target_resolution`.
//   3. The full-mesh AABB tree is used for distance/visibility in every local
//      grid, so neighbouring geometry still blocks leaks correctly.
//
// A local free-space component that touches the local box boundary is the
// surrounding region (room air or true exterior) and is ignored; a component
// fully enclosed by the shell becomes a bounded volume. A global volume id is
// allocated lazily the first time a shell face actually maps to a component, so
// regions that no shell face bounds never produce spurious empty volumes.

struct MultiScaleParams
{
    int target_resolution = 64;       // per-shell grid resolution (shell-relative)
    double clearance_factor = 0.16;
    std::vector<double> probe_factors = {0.18, 0.30, 0.45, 0.60};
    int component_search_radius_cells = 4;
    double bbox_inflate_frac = 0.06;  // pad shell bbox so surrounding air reaches boundary
    // Skip shells whose bounding-box diagonal is below this fraction of the
    // whole-model bounding-box diagonal. Filters out tiny stray fragments
    // (loose triangles, decorative slivers) that cannot enclose a meaningful
    // volume. Set to 0 to disable filtering.
    double min_shell_diag_frac = 0.01;
};

static std::vector<std::vector<std::size_t>> extract_face_shells(
    const std::vector<ObjFace>& faces)
{
    std::map<UndirectedEdge, std::vector<std::size_t>> edge_to_faces;
    for (std::size_t fid = 0; fid < faces.size(); ++fid)
    {
        const auto& vi = faces[fid].vertex_indices;
        const std::size_t n = vi.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            edge_to_faces[UndirectedEdge(vi[i], vi[(i + 1) % n])].push_back(fid);
        }
    }

    std::vector<char> visited(faces.size(), 0);
    std::vector<std::vector<std::size_t>> shells;

    for (std::size_t start = 0; start < faces.size(); ++start)
    {
        if (visited[start])
        {
            continue;
        }

        std::vector<std::size_t> shell;
        std::queue<std::size_t> q;
        q.push(start);
        visited[start] = 1;

        while (!q.empty())
        {
            const std::size_t f = q.front();
            q.pop();
            shell.push_back(f);

            const auto& vi = faces[f].vertex_indices;
            const std::size_t n = vi.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const UndirectedEdge e(vi[i], vi[(i + 1) % n]);
                for (std::size_t nf : edge_to_faces[e])
                {
                    if (!visited[nf])
                    {
                        visited[nf] = 1;
                        q.push(nf);
                    }
                }
            }
        }

        shells.push_back(std::move(shell));
    }

    return shells;
}

static Bbox_3 compute_shell_bbox(
    const std::vector<std::size_t>& shell,
    const std::vector<ObjFace>& faces,
    const std::vector<Point_3>& vertices)
{
    bool first = true;
    Bbox_3 box;
    for (std::size_t fid : shell)
    {
        for (std::size_t vi : faces[fid].vertex_indices)
        {
            if (first)
            {
                box = vertices[vi].bbox();
                first = false;
            }
            else
            {
                box = box + vertices[vi].bbox();
            }
        }
    }
    require(!first, "Empty shell has no bounding box.");
    return box;
}

static Bbox_3 inflate_bbox(const Bbox_3& b, double frac)
{
    const double dx = b.xmax() - b.xmin();
    const double dy = b.ymax() - b.ymin();
    const double dz = b.zmax() - b.zmin();
    const double pad = frac * std::max({dx, dy, dz});
    return Bbox_3(
        b.xmin() - pad, b.ymin() - pad, b.zmin() - pad,
        b.xmax() + pad, b.ymax() + pad, b.zmax() + pad);
}

static double bbox_diagonal(const Bbox_3& b)
{
    const double dx = b.xmax() - b.xmin();
    const double dy = b.ymax() - b.ymin();
    const double dz = b.zmax() - b.zmin();
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static VolumeMappingResult detect_volumes_multiscale(
    const std::vector<ObjFace>& faces,
    const std::vector<Point_3>& vertices,
    const std::vector<OriginalFaceGeometry>& face_geom,
    const Tree& tree,
    const MultiScaleParams& params)
{
    // Index face geometry by original face id for O(1) lookup.
    std::vector<const OriginalFaceGeometry*> geom_by_face(faces.size(), nullptr);
    for (const auto& fg : face_geom)
    {
        require(fg.original_face_id < faces.size(),
                "face_geom id out of range in detect_volumes_multiscale().");
        geom_by_face[fg.original_face_id] = &fg;
    }

    const std::vector<std::vector<std::size_t>> shells = extract_face_shells(faces);
    std::cout << "Detected " << shells.size()
              << " surface shell(s) from mesh connectivity.\n";

    // Whole-model diagonal: used to filter out negligibly small shells.
    const double model_diag = bbox_diagonal(compute_bbox(vertices));
    const double min_shell_diag = params.min_shell_diag_frac * model_diag;

    // face id -> (volume id -> orientation sign)
    std::vector<std::map<std::size_t, int>> face_vol_sign(faces.size());
    int global_volume_id = 0;

    for (std::size_t shell_idx = 0; shell_idx < shells.size(); ++shell_idx)
    {
        const std::vector<std::size_t>& shell = shells[shell_idx];
        try
        {
            const Bbox_3 raw_bbox = compute_shell_bbox(shell, faces, vertices);

            if (min_shell_diag > 0.0 && bbox_diagonal(raw_bbox) < min_shell_diag)
            {
                std::cerr << "  [shell " << shell_idx
                          << " skipped: below min size ("
                          << shell.size() << " face(s))]\n";
                continue;
            }

            const Bbox_3 bbox =
                inflate_bbox(raw_bbox, params.bbox_inflate_frac);

            Grid grid = build_free_space_grid(
                bbox, tree, params.target_resolution, params.clearance_factor);
            label_free_space_components(grid, tree);

            // local component id -> global volume id (allocated lazily)
            std::unordered_map<int, int> local_to_global;

            auto resolve_volume = [&](int comp) -> int
            {
                auto it = local_to_global.find(comp);
                if (it != local_to_global.end())
                {
                    return it->second;
                }
                const int vid = global_volume_id++;
                local_to_global[comp] = vid;
                return vid;
            };

            for (std::size_t fid : shell)
            {
                const OriginalFaceGeometry* fgp = geom_by_face[fid];
                if (fgp == nullptr)
                {
                    continue;
                }
                const OriginalFaceGeometry& fg = *fgp;

                for (const Point_3& base : fg.interior_probe_points)
                {
                    for (double df : params.probe_factors)
                    {
                        const double probe_dist = df * grid.step;
                        const Point_3 p_plus  = base + probe_dist * fg.normal;
                        const Point_3 p_minus = base - probe_dist * fg.normal;

                        const int cp = find_nearby_component_visible(
                            grid, tree, p_plus, params.component_search_radius_cells);
                        const int cm = find_nearby_component_visible(
                            grid, tree, p_minus, params.component_search_radius_cells);

                        if (cp >= 0 && !grid.is_exterior_comp[cp])
                        {
                            // probe along +normal hit it -> normal points INTO volume
                            face_vol_sign[fid].emplace(
                                static_cast<std::size_t>(resolve_volume(cp)), -1);
                        }
                        if (cm >= 0 && !grid.is_exterior_comp[cm])
                        {
                            // probe along -normal hit it -> normal points OUT of volume
                            face_vol_sign[fid].emplace(
                                static_cast<std::size_t>(resolve_volume(cm)), +1);
                        }
                    }
                }
            }
        }
        catch (const std::exception& ex)
        {
            // Open/degenerate/too-thin shells cannot enclose a volume; skip them
            // rather than aborting the whole run.
            std::cerr << "  [shell " << shell_idx << " skipped: " << ex.what() << "]\n";
            continue;
        }
    }

    VolumeMappingResult out;
    out.bounded_volume_count = static_cast<std::size_t>(global_volume_id);
    out.volume_to_original_faces.resize(out.bounded_volume_count);
    out.volume_to_face_signs.resize(out.bounded_volume_count);
    out.original_face_to_volumes.assign(faces.size(), {});

    for (std::size_t fid = 0; fid < faces.size(); ++fid)
    {
        // std::map iterates in ascending volume-id order -> sorted, de-duplicated.
        for (const auto& kv : face_vol_sign[fid])
        {
            const std::size_t vid = kv.first;
            out.original_face_to_volumes[fid].push_back(vid);
            out.volume_to_original_faces[vid].push_back(fid);
            out.volume_to_face_signs[vid].push_back(kv.second);
        }
    }

    return out;
}

static void print_result(const VolumeMappingResult& r)
{
    std::cout << "Bounded volume count: " << r.bounded_volume_count << "\n\n";

    std::cout << "=== Volume -> Original Faces ===\n";
    for (std::size_t vid = 0; vid < r.volume_to_original_faces.size(); ++vid)
    {
        std::cout << "Volume " << vid << ": ";
        for (std::size_t fid : r.volume_to_original_faces[vid])
        {
            std::cout << fid << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n=== Original Face -> Volumes ===\n";
    for (std::size_t fid = 0; fid < r.original_face_to_volumes.size(); ++fid)
    {
        std::cout << "Face " << fid << ": ";
        if (r.original_face_to_volumes[fid].empty())
        {
            std::cout << "(no bounded volume adjacency)";
        }
        else
        {
            for (std::size_t vid : r.original_face_to_volumes[fid])
            {
                std::cout << vid << " ";
            }
        }
        std::cout << "\n";
    }
}

static VolumeMappingResult run_volume_mapping_pipeline(
    const std::string& obj_path,
    int grid_resolution = 48,
    double clearance_factor = 0.22,
    double face_probe_distance_factor = 0.60)
{
    std::vector<Point_3> vertices;
    std::vector<ObjFace> faces;
    read_obj(obj_path, vertices, faces);

    std::vector<Triangle_3> triangles;
    std::vector<TriMeta> tri_meta;
    std::vector<OriginalFaceGeometry> face_geom;
    triangulate_faces(vertices, faces, triangles, tri_meta, face_geom);

    require(!triangles.empty(), "No triangles produced from OBJ faces.");

    Tree tree(triangles.begin(), triangles.end());
    tree.accelerate_distance_queries();

    const Bbox_3 bbox = compute_bbox(vertices);

    Grid grid = build_free_space_grid(bbox, tree, grid_resolution, clearance_factor);
    label_free_space_components(grid, tree);

    VolumeMappingResult result =
        map_faces_to_bounded_volumes(face_geom, grid, tree, {face_probe_distance_factor}, 4);

    result.volume_is_manifold = detect_volume_manifoldness(result, faces);

    return result;
}

// ------------------------------
// JSON export helpers
// ------------------------------

static void write_json_string(std::ostream& os, const std::string& s)
{
    os << '"';
    for (char c : s)
    {
        switch (c)
        {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\b': os << "\\b";  break;
            case '\f': os << "\\f";  break;
            case '\n': os << "\\n";  break;
            case '\r': os << "\\r";  break;
            case '\t': os << "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20)
                {
                    os << "\\u"
                       << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(static_cast<unsigned char>(c))
                       << std::dec << std::setfill(' ');
                }
                else
                {
                    os << c;
                }
                break;
        }
    }
    os << '"';
}

static void write_point_json(std::ostream& os, const Point_3& p)
{
    os << "["
       << std::setprecision(17) << p.x() << ", "
       << std::setprecision(17) << p.y() << ", "
       << std::setprecision(17) << p.z() << "]";
}

static void write_size_t_array_json(std::ostream& os, const std::vector<std::size_t>& arr)
{
    os << "[";
    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        if (i > 0) os << ", ";
        os << arr[i];
    }
    os << "]";
}

static void write_face_vertices_json(
    std::ostream& os,
    const ObjFace& face,
    const std::vector<Point_3>& vertices)
{
    os << "[";
    for (std::size_t i = 0; i < face.vertex_indices.size(); ++i)
    {
        if (i > 0) os << ", ";
        write_point_json(os, vertices[face.vertex_indices[i]]);
    }
    os << "]";
}

static void export_result_to_json(
    const std::string& output_path,
    const VolumeMappingResult& result,
    const std::vector<ObjFace>& faces,
    const std::vector<Point_3>& vertices)
{
    require(result.original_face_to_volumes.size() == faces.size(),
            "Mismatch: original_face_to_volumes size does not match number of faces.");

    std::ofstream out(output_path);
    require(out.good(), "Could not open JSON output file: " + output_path);

    out << "{\n";
    out << "  \"bounded_volume_count\": " << result.bounded_volume_count << ",\n";

    out << "  \"volumes\": [\n";
    for (std::size_t vid = 0; vid < result.volume_to_original_faces.size(); ++vid)
    {
        if (vid > 0) out << ",\n";

        out << "    {\n";
        out << "      \"volume_id\": " << vid << ",\n";
        out << "      \"is_manifold\": " << ((vid < result.volume_is_manifold.size() && result.volume_is_manifold[vid]) ? "true" : "false") << ",\n";
        out << "      \"faces\": [\n";

        const auto& face_ids = result.volume_to_original_faces[vid];
        for (std::size_t i = 0; i < face_ids.size(); ++i)
        {
            if (i > 0) out << ",\n";

            const std::size_t fid = face_ids[i];
            require(fid < faces.size(), "Face id out of range while writing JSON.");

            const int sign = (vid < result.volume_to_face_signs.size() &&
                              i < result.volume_to_face_signs[vid].size())
                             ? result.volume_to_face_signs[vid][i] : 1;

            out << "        {\n";
            out << "          \"face_id\": " << fid << ",\n";
            out << "          \"sign\": " << sign << ",\n";
            out << "          \"volumes\": ";
            write_size_t_array_json(out, result.original_face_to_volumes[fid]);
            out << ",\n";
            out << "          \"vertices\": ";
            write_face_vertices_json(out, faces[fid], vertices);
            out << "\n";
            out << "        }";
        }

        out << "\n";
        out << "      ]\n";
        out << "    }";
    }
    out << "\n";
    out << "  ],\n";

    out << "  \"original_face_to_volumes\": [\n";
    for (std::size_t fid = 0; fid < result.original_face_to_volumes.size(); ++fid)
    {
        if (fid > 0) out << ",\n";

        out << "    {\n";
        out << "      \"face_id\": " << fid << ",\n";
        out << "      \"volumes\": ";
        write_size_t_array_json(out, result.original_face_to_volumes[fid]);
        out << ",\n";
        out << "      \"vertices\": ";
        write_face_vertices_json(out, faces[fid], vertices);
        out << "\n";
        out << "    }";
    }
    out << "\n";
    out << "  ]\n";

    out << "}\n";
}

// ------------------------------

static void export_volume_to_obj(
    const std::string& folder,
    std::size_t volume_id,
    const std::vector<std::size_t>& face_ids,
    const std::vector<ObjFace>& all_faces,
    const std::vector<Point_3>& all_vertices)
{
    std::filesystem::create_directories(folder);

    std::string filename = folder + "/volume_" + std::to_string(volume_id) + ".obj";
    std::ofstream out(filename);
    require(out.good(), "Could not open OBJ output file: " + filename);

    // Collect all unique vertices used in this volume's faces
    std::set<std::size_t> vertex_set;
    for (std::size_t fid : face_ids) {
        require(fid < all_faces.size(), "Face id out of range in export_volume_to_obj.");
        const ObjFace& face = all_faces[fid];
        for (std::size_t vi : face.vertex_indices) {
            vertex_set.insert(vi);
        }
    }

    // Assign new indices to vertices (OBJ 1-based)
    std::map<std::size_t, std::size_t> vertex_index_map;
    std::size_t new_idx = 1;
    for (std::size_t old_vi : vertex_set) {
        vertex_index_map[old_vi] = new_idx++;
    }

    // Write vertices
    out << "# Volume " << volume_id << " OBJ export\n";
    out << "# Original faces: ";
    for (std::size_t i = 0; i < face_ids.size(); ++i) {
        if (i > 0) out << ", ";
        out << face_ids[i];
    }
    out << "\n";
    out << std::setprecision(17);

    for (std::size_t old_vi : vertex_set) {
        const Point_3& p = all_vertices[old_vi];
        out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }

    // Write faces
    for (std::size_t fid : face_ids) {
        const ObjFace& face = all_faces[fid];
        out << "f";
        for (std::size_t old_vi : face.vertex_indices) {
            std::size_t new_vi = vertex_index_map.at(old_vi);
            out << " " << new_vi;
        }
        out << "\n";
    }
}

// ------------------------------

int main(int argc, char** argv)
{
    try
    {
        require(argc >= 2,
                "Usage: ./volume_detector input.(json|obj) [output.json]");

        const std::string input_path = argv[1];
        const std::string json_path  = (argc >= 3) ? argv[2] : "volume_faces.json";

        std::vector<Point_3> vertices;
        std::vector<ObjFace> faces;
        read_mesh(input_path, vertices, faces);

        std::vector<Triangle_3> triangles;
        std::vector<TriMeta> tri_meta;
        std::vector<OriginalFaceGeometry> face_geom;
        triangulate_faces(vertices, faces, triangles, tri_meta, face_geom);

        require(!triangles.empty(), "No triangles produced from input faces.");

        Tree tree(triangles.begin(), triangles.end());
        tree.accelerate_distance_queries();

        // Multi-scale detection: one shell-relative local grid per connected
        // surface shell, so big rooms and small furniture cavities are both
        // resolved without a globally fine (and globally expensive) grid.
        MultiScaleParams params;
        VolumeMappingResult result =
            detect_volumes_multiscale(faces, vertices, face_geom, tree, params);

        result.volume_is_manifold = detect_volume_manifoldness(result, faces);

        print_result(result);
        export_result_to_json(json_path, result, faces, vertices);

        // Export each volume to OBJ
        const std::string obj_folder = "generated-obj-volume";
        for (std::size_t vid = 0; vid < result.volume_to_original_faces.size(); ++vid) {
            const auto& face_ids = result.volume_to_original_faces[vid];
            export_volume_to_obj(obj_folder, vid, face_ids, faces, vertices);
        }

        std::cout << "\nJSON exported to: " << json_path << "\n";
        std::cout << "OBJ files exported to folder: " << obj_folder << "\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
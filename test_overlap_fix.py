from shapely.geometry import Polygon

# Simulate your edge-adjacent rectangles with 1.15mm gap
face1 = Polygon([(-0.60025, -1.365511), (0.99975, -1.365511), (0.99975, -0.565511), (-0.60025, -0.565511)])
face2 = Polygon([(-2.199095, -1.365511), (-0.599095, -1.365511), (-0.599095, -0.565511), (-2.199095, -0.565511)])

inter = face1.intersection(face2)
print(f"Intersection type: {inter.geom_type}")
print(f"Intersection area: {inter.area:.10f}")
print(f"Intersection length: {inter.length:.10f}")

if inter.area > 0 and inter.length > 0:
    ratio = inter.area / inter.length
    print(f"Area/Perimeter ratio: {ratio:.10f}")
    print(f"Passes 0.001 threshold: {ratio >= 0.001}")
else:
    print("No area intersection")

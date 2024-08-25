"""
geolib.py
Library with classes to handle vectors and geometry figures easily.

Author
   Julia Wutschka

Usage
   Import this script as a library. It can't be used standalone.

"""
import math

type intersection_result = tuple[bool, Vector]

def next_index_value(current_index_value, index_count):
    """
    Add 1 to a given index value. If the max value reached set index back to 0.

    Helper function to go through arrays with range(index_count) for more loops.
    :param current_index_value: the current value (index)
    :param index_count: the value where to start from 0.
    :returns: *True* if the points describe a valid polygon
    """
    return (current_index_value + 1) % index_count


def is_polygon(list_of_points):
    """
    Checks if a list of points represents a polygon

    :param list_of_points: a list() object containing `Vector`objects which represents the corners of the polygon
    :returns: *True* if the points describe a valid polygon
    """
    if len(list_of_points) < 3:
        return False
    edge_count = 0
    points = []
    straights = []
    for i in range(len(list_of_points)):
        new_straight = Straight(list_of_points[i],
                                list_of_points[(i + 1) % len(list_of_points)])
        if edge_count > 0 and straights[-1].is_identical(new_straight):
            if straights[-1].base.is_same(new_straight.base2):
                return False
            straights[-1] = Straight(straights[-1].base, new_straight.base2)
        else:
            straights.append(new_straight)
            points.append(list_of_points[i])
            edge_count += 1
    if edge_count > 1 and straights[-1].is_identical(straights[0]):
        straights[0] = Straight(straights[-1].base, straights[0].base2)
        edge_count -= 1
        points.pop()
        straights.pop()
    if edge_count < 3:
        return False
    return True


def sign(x):
    """
    Sign function

    See https://en.wikipedia.org/wiki/Sign_function

    :param x: numeric value
    :returns: 1 if x is > 0, -1 if x < 0, otherwise 0
    """
    return 1 if x > 0 else (-1 if x < 0 else 0)


def angel_diff(alpha, beta):
    delta = (alpha - beta + 2 * math.pi) % (2 * math.pi)
    return (2 * math.pi - delta) if delta > math.pi else delta


class SimpleVector:
    """
    A simple vector representation class ony used for default vector values.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0):
        """
        Constructor
        If absolut is set the vector the values for phi and absolut will be used

        :param x: the x-value of the Vector
        :param y: the y-value of the Vector
        """
        self.x = x
        self.y = y

    def __str__(self):
        return f"geolib.Vector({self.x}, {self.y})"


class Vector(SimpleVector):
    """
    Class for 2-dimensional vectors. Should be used for describing points on a plane
    """

    def __init__(self, x: float = 0, y: float = 0, phi: float = 0.0, absolut: float = 0.0):
        """
        Constructor

        :param x: the x-value of the Vector
        :param y: the y-value of the Vector
        :param phi: angel (if polar coordinates are used)
        :param absolut: the absolut value (if polar coordinates are used)
        """
        if absolut != 0.0:
            super().__init__(math.cos(phi) * absolut, math.sin(phi) * absolut)
            self.has_polar = True
            self.orientation = phi
        else:
            super().__init__(x, y)
            self.has_polar = False
            self.orientation = 0

    def get_orientation(self) -> float:
        if self.has_polar:
            return self.orientation
        self.orientation = math.acos(self.x / self.abs())
        if self.y < 0:
            self.orientation = 2 * math.pi - self.orientation
        self.has_polar = True
        return self.orientation

    def is_same(self, vector):
        """
        Comparison of two vectors

        :param vector: a Vector-Class object to be compared with
        :returns: *True* if both vectors are identical
        """
        return self.x == vector.x and self.y == vector.y

    def add(self, vector):
        """
        Add a vector and returns a new vector object. The original vector will be unchanged.

        :param vector: a Vector-Class object to be added
        :returns: new vector, value = self + vector
        """
        return Vector(self.x + vector.x, self.y + vector.y)

    def sub(self, vector):
        """
        Substracts a vector and returns a new vector object. The original vector will be unchanged.

        :param vector: a Vector-Class object which will be substracted
        :returns: new vector, value = self - vector
        """
        return Vector(self.x - vector.x, self.y - vector.y)

    def s_mul(self, s):
        """
        Multiplies the vector with a scalar and returns a new vector object. The original vector will be unchanged.

        :param s: the scalar with which the vector will be multiplied
        :returns: new vector, value = self * s
        """
        return Vector(self.x * s, self.y * s)

    def m_mul(self, matrix):
        """
        Calculates the scalar product with the vector

        :param matrix: a matrix-Class object with which the vector will be multiplied
        :returns: new vector, value = self * matrix
        """
        return Vector(matrix.a1 * self.x + matrix.a2 * self.y,
                      matrix.b1 * self.x + matrix.b2 * self.y, )

    def scalar_product(self, vector):
        """
        Multiplies the vector with a 2x2-matrix and returns a new vector object. The original vector will be unchanged.

        :param vector: a vector-Class object with which the vector will be multiplied
        :returns: new vector, value = self * matrix
        """
        return self.x * vector.x + self.y * vector.y

    def abs2(self) -> float:
        """
        Calculates the square of the absolute value

        :returns: |self| ^2
        """
        return self.x * self.x + self.y * self.y

    def abs(self) -> float:
        """
        Calculates the absolute value

        :returns: |self|
        """
        return math.sqrt(self.abs2())

    def linearly_dependency(self, vector):
        """
        Calculate the relation of the absolute value to a given vector.

        |s| is the absolute value of this vector and |v| the value of the parameter value.
        The function will return |s|/|v| if both vectors are linearly dependent otherwise 0.
        If v is the inverse vector of s the result is -1.

        :param vector: the vector v
        :returns: 0, if s and v are linearly independent, otherwise |s|/|v|
        """
        if vector.y == 0:
            if self.y == 0:
                if vector.x == 0:
                    return 1.0
                else:
                    return self.x / vector.x
            else:
                return 0
        vx = vector.x * self.y / vector.y
        if round(vx, 6) != round(self.x, 6):
            return 0
        if vector.x == 0:
            return self.y / vector.y
        return self.x / vector.x

    def is_linearly_dependent(self, vector):
        """
        Check if a vector linearly independent to the vector object.


        :param vector: the vector v
        :returns: *True*, if s and v are linearly independent
        """
        return self.linearly_dependency(vector) != 0

    def rotate_cs(self, center, aspect, sn, cs):
        """
        Rotates and resizes the vector

        Rotates the vector by the angel a using the values sin(a) and cos(a) around a center point *center*
        and rescale it by the factor *aspect*.
        Returns a new vector object. The original vector will be unchanged.
        :param center: the center point of the rotation
        :param aspect: the factor the vector will be stretched
        :param sn: the sine-value of the rotation angel
        :param cs: the cosine-value of the rotation angel
        :returns: new vector object
        """
        rotation_matrix = Matrix(cs, -sn,
                                 sn, cs)
        return self.sub(center).m_mul(rotation_matrix).s_mul(aspect).add(center)

    def rotate(self, angel_rad, center=SimpleVector(0.0, 0.0), aspect=1.0):
        """
        Rotates and resizes the vector

        Rotates the vector by the angel angel_rad around a center point *center*
        and rescale it by the factor *aspect*.
        Returns a new vector object. The original vector will be unchanged.
        :param angel_rad: rotation angel in radians (counterclockwise)
        :param center: the center point of the rotation. Default value is (0, 0)
        :param aspect: the factor the vector will be stretched. Default value is 1.0
        :returns: new vector object
        """
        return self.rotate_cs(center, aspect, math.sin(angel_rad), math.cos(angel_rad))

    def rotate90(self, center=SimpleVector(0.0, 0.0), aspect=1.0):
        """
        Rotates and resizes the vector counterclockwise by 90°

        Rotates the vector by 90° CCW around a center point *center*
        and rescale it by the factor *aspect*.
        Returns a new vector object. The original vector will be unchanged.
        :param center: the center point of the rotation. Default value is (0, 0)
        :param aspect: the factor the vector will be stretched. Default value is 1.0
        :returns: new vector object
        """
        return self.rotate_cs(center, aspect, 1.0, 0.0)

    def rotate270(self, center=SimpleVector(0.0, 0.0), aspect=1.0):
        """
        Rotates and resizes the vector counterclockwise by 90°

        Rotates the vector by 90° CW (270°CCW) around a center point *center*
        and rescale it by the factor *aspect*.
        Returns a new vector object. The original vector will be unchanged.
        :param center: the center point of the rotation. Default value is (0, 0)
        :param aspect: the factor the vector will be stretched. Default value is 1.0
        :returns: new vector object
        """
        return self.rotate_cs(center, aspect, -1.0, 0.0)


class Matrix:
    """
    Class for 2x2-dimensional matrices. Usually used for vector-transformations.
    """

    def __init__(self, a1, a2, b1, b2) -> None:
        """
        Constructor for matrix
        | a1 a2 |
        | b1 b2 |

        :param a1: a1 (float)
        :param a2: a2 (float)
        :param b1: b1 (float)
        :param b2: b21 (float)
        """
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2

    def __str__(self) -> str:
        return f"({self.a1},{self.a2},{self.b1},{self.b2})"

    def s_mul(self, s):
        """
        Multiplies the matrix with a scalar and returns a new matrix object. The original matrix will be unchanged.

        :param s: the scalar with which the matrix will be multiplied
        :returns: new matrix, value = self * s
        """
        return Matrix(self.a1 * s, self.a2 * s, self.b1 * s, self.b2 * s)

    def v_mul(self, vector: Vector):
        """
        Multiplies the matrix with a vector and returns a new vector object. The original matrix will be unchanged.

        :param vector: the vector with which the matrix will be multiplied
        :returns: new vector, value = self * vector
        """
        return Vector(self.a1 * vector.x + self.a2 * vector.y, self.b1 * vector.x + self.b2 * vector.y)

    def m_mul(self, matrix):
        return Matrix(
            self.a1 * matrix.a1 + self.a2 * matrix.b1, self.a1 * matrix.a2 + self.a2 * matrix.b2,
            self.b1 * matrix.a1 + self.b2 * matrix.b1, self.b1 * matrix.a2 + self.b2 * matrix.b2)


class Straight:
    def __init__(self, start_point: Vector = Vector(0, 0), point_b: Vector = None,
                 length_ac=0, length_cb=0, angel_rad=0, ) -> None:
        """
        Constructor
        alternative 1: construct a straight from start_point to point_b
        alternative 2: start_point ("point C" = center) is a point on the straight A-B
            a straight will be defined by the length AC, length CB and the angel of the straight
        :param start_point: Point A or center point if point_b ist not given
        :param point_b: Point b, if given a straight between point A and point B is defined
            have to be none for the construction using alternative 2
        :param length_ac: length AC
        :param length_cb: length CB
        :param angel_rad: angel AB to x-axis
        """
        if point_b is None:
            if length_ac + length_cb == 0:
                raise Exception("Could not create straight with length 0 from identical points")
            self.base = start_point.sub(Vector(1, 0).rotate(angel_rad, aspect=length_ac))
            self.base2 = start_point.add(Vector(1, 0).rotate(angel_rad, aspect=length_cb))
        else:
            if start_point.is_same(point_b):
                raise Exception("Could not create straight from identical points")
            self.base = start_point
            self.base2 = point_b
        self.direction = self.base2.sub(self.base)
        self.length = self.direction.abs()
        self.half_pi_left = self.direction.rotate90(Vector(0, 0), 1.0 / self.length)

    def __str__(self) -> str:
        return f"geolib.Straight({self.base}, {self.base2} )  # Direction {self.direction}"

    def has_same_slope(self, straight_2):
        """
        Checks is the given straight have the same slope
        :param straight_2: straight to be compared with
        :return: True, if the straights have the same slopes
        """
        return self.direction.is_linearly_dependent(straight_2.direction)

    def orthogonal_projection(self, vector):
        """

        :param vector: point
        :return: point on the straight representing the orthogonal projection
        """
        return self.base.add(self.direction.s_mul(self.direction.scalar_product(vector.sub(self.base)) /
                                                  self.direction.abs2()))

    def get_orientation(self) -> float:
        """
        Get the angel in relation to the x-axis
        :return: orientation/slope in radix
        """
        return self.direction.get_orientation()

    def get_side(self, vector):
        """

        :param vector:
        :return:
        """
        # 1 if on the left side, -1 right side, 0 on straight
        return sign(vector.sub(self.orthogonal_projection(vector)).linearly_dependency(self.half_pi_left))

    def get_distance(self, vector):
        # Berechnet den Abstand zur GERADEN point_a, point_b
        return self.orthogonal_projection(vector).sub(vector).abs()

    def get_inner_distance(self, vector):
        # Berechnet den Abstand zur STRECKE point_a, point_b
        s = self.orthogonal_projection(vector)
        if self.is_on_segment(s):
            return s.sub(vector).abs()
        else:
            return min(Straight(self.base, vector).length, Straight(self.base2, vector).length)

    def get_intersection(self, straight_2):
        if self.has_same_slope(straight_2):
            return None
        if self.direction.y == 0:
            s = - (straight_2.base.y - self.base.y) / straight_2.direction.y
        else:
            s = ((straight_2.base.x - self.base.x) - (
                    straight_2.base.y - self.base.y) * self.direction.x / self.direction.y) / \
                (self.direction.x / self.direction.y * straight_2.direction.y - straight_2.direction.x)
        return Vector(straight_2.base.x + s * straight_2.direction.x,
                      straight_2.base.y + s * straight_2.direction.y)

    def has_intersection_on_segment(self, straight_2):
        s = self.get_intersection(straight_2)
        if s is None:
            return False
        return self.is_on_segment(s)

    def has_intersection_on_both_segments(self, straight_2):
        s = self.get_intersection(straight_2)
        if s is None:
            return False
        return self.is_on_segment(s) and straight_2.is_on_segment(s)

    def has_intersection_on_both_segments_v2(self, straight_2) -> intersection_result:
        """

        :param straight_2:
        :return: tuple(True is the intersection in on both segments, intersection point)
        """
        s = self.get_intersection(straight_2)
        if s is None:
            return False, Vector(0, 0)
        return self.is_on_segment(s) and straight_2.is_on_segment(s), s

    def is_on_straight(self, vector_to_compare):
        return self.direction.linearly_dependency(vector_to_compare.sub(self.base)) != 0

    def is_on_ray(self, vector_to_compare):
        return self.direction.linearly_dependency(vector_to_compare.sub(self.base)) > 0

    def is_on_segment(self, vector_to_compare):
        return self.direction.linearly_dependency(vector_to_compare.sub(self.base)) > 1

    def is_parallel(self, straight_to_compare):
        return self.direction.is_linearly_dependent(straight_to_compare.direction)

    def is_identical(self, straight_2):
        r1 = self.direction.is_linearly_dependent(straight_2.direction)
        r2 = self.is_on_straight(straight_2.base)
        return r1 and r2

    def rotate(self, angel_rad, center=SimpleVector(0.0, 0.0), aspect=1.0):
        """
        Rotates and resizes a straight

        Rotates the straight by the angel angel_rad around a center point *center*
        and rescale it by the factor *aspect*.
        Returns a new straight object. The original straight will be unchanged.
        :param angel_rad: rotation angel in radians (counterclockwise)
        :param center: the center point of the rotation. Default value is (0, 0)
        :param aspect: the factor the straight will be stretched. Default value is 1.0
        :returns: new straight object
        """
        sn = math.sin(angel_rad)
        cs = math.cos(angel_rad)
        return Straight(self.base.rotate_cs(center, aspect, sn, cs),
                        self.base2.rotate_cs(center, aspect, sn, cs))

    def get_perpendicular_bisector(self):
        """
        Returns the perpendicular bisector to the segment of the base points

        Perpendicular bisector = DE: Mittelsenkrechte
        :returns: new straight object which represents the perpendicular bisector
        """
        return Straight(self.base.add(self.direction.s_mul(0.5)),
                        self.base.add(self.direction.s_mul(0.5)).add(self.half_pi_left))

    def move(self, direction_vector):
        """

        :param direction_vector:
        :return:
        """
        return Straight(self.base.add(direction_vector), self.base2.add(direction_vector))

class GeoFigure:
    """
    Abstract class for figures. Needs to bei inherited.
    """

    def __init__(self) -> None:
        self.frame_min = Vector()
        self.frame_max = Vector()

    def __str__(self) -> str:
        return "Abstract geometry figure"

    def is_inside(self, vector) -> bool:
        """
        Checks, if a point inside the figure.
        :param vector: vector-object, representing the point to test
        :return: true, if inside
        """
        return False

    def get_distance(self, vector) -> float:
        """
        Get the nearest distance from a point to the figure's border
        :param vector: vector-object, representing the point to test
        :return: the distance, 0 if the point is inside the figure
        """
        return 0.0

    def is_in_polygon(self, outer_polygon) -> bool:
        """
        Checks, if a polygon is inside the figure. Therefore, all points of the polygon are inside the figure and there
        are no intersections of both borderlines
        :param outer_polygon: Polygon-object to be tested
        :return: true, if inside, else false
        """
        return False

    def is_convex(self) -> bool:
        """
        Checks, if the figure is convex
        :return: true, if convex, else false (concave)
        """
        return True

    def rotate(self, angel_rad, center=SimpleVector(0.0, 0.0), aspect=1.0):
        """
        Rotates the figure about *angel_rad* around the point *center* and stretches it about *aspect*
        :param angel_rad: angel to rotate (radian value)
        :param center: vector-object, representing the center point of the rotation
        :param aspect: factor to stretch the figure
        :return: new object of the rotated figure
        """
        return GeoFigure()

    def get_border(self, distance):
        """
        Get a figure which represents a border around the figure
        :param distance: distance of the border
        :return: new object representing the border
        """
        return GeoFigure()

    def move(self, direction_vector):
        """
        Move the figure by vector *direction_vector*
        :param direction_vector: vector-object, representing the movement
        :return: new object of the moved figure
        """
        return GeoFigure()

class Polygon(GeoFigure):
    """
    Class for N-polygon which could be also concave. Intersection of borderline may lead to unpredictable results.
    """
    def __init__(self, list_of_points, check_polygon=True) -> None:
        """
        Constructor of a N-polgon. Points lying on an edge will be ignored.
        :param list_of_points: list of vector-objects representing the vertexes
        :param check_polygon: if true, a check will be performed. if the check fails a exception is raised.
        """
        super().__init__()
        if check_polygon:
            if not (is_polygon(list_of_points)):
                raise Exception("Polygon - List of points do not define a polygon")
        self.edge_count = 0
        self.points = []
        self.straights = []
        self.orientations = []
        # orientation[x] is 1 if straight[x+1] is on the left side of straight[x]
        for i in range(len(list_of_points)):
            new_straight = Straight(list_of_points[i],
                                    list_of_points[(i + 1) % len(list_of_points)])
            if self.edge_count > 0 and self.straights[-1].is_identical(new_straight):
                self.straights[-1] = Straight(self.straights[-1].base, new_straight.base2)
            else:
                self.straights.append(new_straight)
                self.points.append(list_of_points[i])
                self.edge_count += 1
        if self.edge_count > 1 and self.straights[-1].is_identical(self.straights[0]):
            self.straights[-1] = Straight(self.straights[-1].base, self.straights[0].base2)
            self.edge_count -= 1
            self.points.pop(0)
            self.straights.pop(0)
        for i in range(self.edge_count):
            self.orientations.append(self.straights[i].get_side(self.straights[(i + 1) % self.edge_count].base2))
        self.orientation = sign(sum(self.orientations))
        self.convex = self.is_convex()
        # orientation is 1 if order of points is against the clock otherwise orientation is -1
        self.frame_min = Vector(min(self.points, key=lambda vector: vector.x).x,
                                min(self.points, key=lambda vector: vector.y).y)
        self.frame_max = Vector(max(self.points, key=lambda vector: vector.x).x,
                                max(self.points, key=lambda vector: vector.y).y)

    def __str__(self) -> str:
        return f"{self.edge_count}-gon([" + ', '.join(str(p) for p in self.points) + "]"

    def is_inside(self, vector):
        if (vector.x < self.frame_min.x and vector.y < self.frame_min.y or
                vector.x > self.frame_max.x and vector.y > self.frame_max.y):
            return False
        # testtrace is a horizontal ray to the right
        testtrace = Straight(vector, vector.add(Vector(1, 0)))
        intersection_count = 0
        test_straights = list()
        # remove parallels to testtrace from polygon
        for s in self.straights:
            if s.direction.y != 0:
                test_straights.append(s)
        i = 0
        while i < len(test_straights):
            s = test_straights[i].get_intersection(testtrace)
            if not (s is None):
                if testtrace.is_on_ray(s):
                    if test_straights[i].base2.is_same(s):
                        if (test_straights[i].direction.y / test_straights[
                                next_index_value(i, len(test_straights))].direction.y) > 0:
                            intersection_count += 1
                    elif test_straights[i].is_on_segment(s):
                        intersection_count += 1
            i += 1
        return (intersection_count % 2) == 1

    def count_intersection(self, straight):
        """
        Counts the intersection of the segment (!) with the polygon
        :param straight: straight-object representing the straight to test
        :return: number of intersections count
        """
        result = 0
        for s in self.straights:
            result +=  1 if s.has_intersection_on_both_segments(straight) else 0
        return result

    def get_distance(self, vector):
        distances_list = list()
        if self.is_inside(vector):
            return 0
        for examinedStraight in self.straights:
            distance = examinedStraight.get_inner_distance(vector)
            if distance > 0:
                distances_list.append(distance)
            return min(distances_list)

    def is_self_intersecting(self):
        """
        Checks, if the edges of the polygon intersect themselves
        :return: true, if the polygon is self intersecting
        """
        res_is_self_intersecting = False
        for i in range(self.edge_count - 1):
            for j in range(i + 1, self.edge_count):
                res_is_self_intersecting = \
                    res_is_self_intersecting and \
                    self.straights[i].has_intersection_on_both_segments(self.straights[j])
        return res_is_self_intersecting

    def is_in_polygon(self, outer_polygon):
        is_outside = True
        for my_point in self.points:
            is_outside = is_outside and outer_polygon.is_inside(my_point)
        if is_outside:
            for my_straight in self.straights:
                for outer_straight in outer_polygon.straights:
                    is_outside = is_outside and not (my_straight.has_intersection_on_both_segments(outer_straight))
        return is_outside

    def is_convex(self):
        res = True
        for o in self.orientations:
            res = res and self.orientation == o
        return res

    def rotate(self, angel_rad, center=SimpleVector(0.0, 0.0), aspect=1.0):
        new_points = list()
        for i in range(self.edge_count):
            new_points.append(self.points[i].rotate(angel_rad, center, aspect))
        return Polygon(new_points, False)

    def get_border(self, distance: float):
        border_straights = []
        new_corner_points = []
        for i in range(self.edge_count):
            move_vector = self.straights[i].half_pi_left.s_mul(-distance * self.orientation)
            new_bs = Straight(self.straights[i].base.add(move_vector), self.straights[i].base2.add(move_vector))
            border_straights.append(new_bs)
            if self.orientations[i] == self.orientation:
                # Next corner is convex
                new_point = self.straights[i].half_pi_left.add(self.straights[(i + 1) % self.edge_count].half_pi_left)
                new_point = self.straights[i].base2.add(new_point.s_mul(distance / new_point.abs() * (- self.orientation)))
                new_corner_points.append((i, new_point))
            else:
                new_point = new_bs.base2.add(new_bs.direction.s_mul(0.0001))
                # make the a new border point just next to the corner to connect the border lines
                new_corner_points.append((i, new_point))
        for np in new_corner_points:
            border_straights.append(Straight(border_straights[np[0]].base2, np[1]))
            border_straights.append(Straight(np[1], border_straights[(np[0] + 1) % self.edge_count].base))
        max_y = max(border_straights, key= lambda s: s.base.y).base.y
        min_y = min(border_straights, key= lambda s: s.base.y).base.y
        list.sort(border_straights, key= lambda s: (s.base.x * (max_y - min_y) + s.base.y - min_y))

        sorted_border_list = []
        border_straight_index = 0
        finished = False
        while not finished:
            # Sort border straights so they create a closed polygon
            sorted_border_list.append(border_straights[border_straight_index])
            border_straight_index =  filter(lambda s: (s[1].base.x == border_straights[border_straight_index].base2.x
                                     and s[1].base.y == border_straights[border_straight_index].base2.y),
                          enumerate(border_straights)).__next__()[0]
            finished = border_straight_index == 0
        border_straights = sorted_border_list

        new_points = []
        border_straight_index = 0
        border_straight_count = len(border_straights)
        finished = False
        final_border_straights = []

        while not finished:
            inter_sec_points: []
            for border_straight_test_index in range(border_straight_index + 1, border_straight_index + self.edge_count):
                test_index = border_straight_test_index % border_straight_count
                inter_sec_point = border_straights[border_straight_index].has_intersection_on_both_segments_v2(
                    border_straights[test_index])
                if inter_sec_point[0]:
                    # noinspection PyTypeChecker
                    inter_sec_points.append(
                        (test_index, inter_sec_point[1],
                         border_straights[border_straight_index].direction.linearly_dependency(inter_sec_point[1])))
            if len(inter_sec_points) > 0:
                list.sort(inter_sec_points, key=lambda s: s[2], reverse=True)
                border_straights[border_straight_index] = Straight(border_straights[border_straight_index].base, inter_sec_points[0][1])
                border_straights[inter_sec_points[0][0]] = Straight(inter_sec_points[0][1], border_straights[inter_sec_points[0][0]].base2)
                new_points.append(inter_sec_points[0][1])
                border_straight_index = inter_sec_points[0][0]
            else:
                new_points.append(border_straights[border_straight_index].base2)
                filtered_straights = list(filter(lambda s: (s[1].base.x == border_straights[border_straight_index].base2.x
                                                          and s[1].base.y == border_straights[border_straight_index].base2.y),
                              enumerate(border_straights)))
                border_straight_index = filtered_straights[0][0]
            final_border_straights.append(border_straights[border_straight_index])
            finished = border_straight_index == 0
        # noinspection PyUnreachableCode
        return Polygon(new_points, False)

    # noinspection PyUnreachableCode
    def move(self, direction_vector):
        new_points = []
        for p in self.points:
            new_points.append(p.add(direction_vector))
        return Polygon(new_points)


class Circle(GeoFigure):
    """
    Class for a circle
    """
    def __init__(self, center_point=SimpleVector(0.0, 0.0), radius=0.0, list_of_circle_points=None) -> None:
        """
        Constructor of a circle. The object can be defined by two ways:
        # by center_point and radius
        # by a list of three points
        An exception will be raised if the list is not none and not three points are given or the three points are on one
        straight. An exception will also be raised, if the list is none and the radius is <= 0.
        :param center_point: vector-object representing the center point (ignored if list_of_circle_points is not *None*)
        :param radius: the radius of the circle (ignored if list_of_circle_points is not *None*)
        :param list_of_circle_points: list of three vector-objects representing points on the circle.
        """
        super().__init__()
        if radius > 0:
            self.center_point = Vector(center_point.x, center_point.y)
            self.radius = radius
        else:
            if list_of_circle_points is None:
                raise Exception("Cannot create circle.")
            if len(list_of_circle_points) != 3:
                raise Exception("Need 3 Points (Vector-Objects) to create circle.")
            t1 = Straight(list_of_circle_points[0], list_of_circle_points[1])
            t2 = Straight(list_of_circle_points[1], list_of_circle_points[2])
            if t1.is_parallel(t2):
                raise Exception("Cannot create circle - the 3 Points (Vector-Objects) are on one straight.")
            self.center_point = t1.get_perpendicular_bisector().get_intersection(t2.get_perpendicular_bisector())
            self.radius = Straight(self.center_point, list_of_circle_points[0]).length
        self.frame_max = Vector(self.center_point.x + self.radius, self.center_point.y + self.radius)
        self.frame_min = Vector(self.center_point.x - self.radius, self.center_point.y - self.radius)

    def __str__(self) -> str:
        return "Circle(" + self.center_point.__str__() + "," + self.radius.__str__() + ")"

    def is_inside(self, vector):
        if (vector.x < self.frame_min.x and vector.y < self.frame_min.y or
                vector.x > self.frame_max.x and vector.y > self.frame_max.y):
            return False
        # testtrace is a horizontal ray to the right
        return self.get_distance(vector) <= self.radius

    def get_distance(self, vector):
        return vector.sub(self.center_point).abs()

    def is_in_polygon(self, outer_polygon: Polygon) -> bool:
        is_inside = outer_polygon.is_inside(self.center_point)
        if is_inside:
            for op_straight in outer_polygon.straights:
                is_inside = is_inside and (len(self.get_intersections(op_straight)) == 0)
            is_inside = Straight(self.center_point, outer_polygon.points[0]).length < self.radius
        return is_inside

    def get_intersections(self, intersection_straight: Straight) -> list:
        intersections = list()
        s = intersection_straight.orthogonal_projection(self.center_point)
        straight_s = Straight(self.center_point, s)
        if straight_s.length == self.radius:
            intersections.append(s)
        elif straight_s.length < self.radius:
            d = math.sqrt(self.radius * self.radius - straight_s.length * straight_s.length)
            intersections.append(s.add(straight_s.half_pi_left.s_mul(d)))
            intersections.append(s.add(straight_s.half_pi_left.s_mul(-d)))
        return intersections

    def get_tangents(self, vector):
        """
        Get a list of tangents according to the given point. If is the point inside the circle, the list is empty.
        If is the point on the cicrle line the result list contains one straight object otherwise two.
        :param vector: vector-object representing the point which is on the tangens
        :return: list of straight objects
        """
        distance = self.get_distance(vector)
        list_of_tangents = []
        if distance == self.radius:
            list_of_tangents.append(Straight(vector, vector.add(Straight(vector, self.center_point).half_pi_left)))
        elif distance > self.radius:
            # https://de.serlo.org/mathe/1647/tangente-an-kreis
            dist_q = Straight(self.center_point, vector).length
            f = self.radius / (dist_q * dist_q)
            s_base = self.center_point.add(vector.sub(self.center_point).s_mul(self.radius * f))
            s_sqrt = Vector(-vector.y + self.center_point.y, vector.x - self.center_point.x).s_mul(
                math.sqrt(dist_q * dist_q - self.radius * self.radius) * f)
            list_of_tangents.append(Straight(vector, s_base.add(s_sqrt)))
            list_of_tangents.append(Straight(vector, s_base.sub(s_sqrt)))
        return list_of_tangents

    def is_convex(self):
        return True

    def rotate(self, angel_rad, center=SimpleVector(0.0, 0.0), aspect=1.0):
        return Circle(self.center_point.rotate(angel_rad, center, aspect), self.radius * aspect)

    def get_border(self, distance):
        return Circle(self.center_point, self.radius + distance)

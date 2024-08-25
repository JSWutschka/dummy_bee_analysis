"""
dummy_class.py
Class definition for the dummy and the feather

Author
   Julia Wutschka

Usage
   This script has to be included.
"""
import geolib
from geolib import SimpleVector
import cv2

color_feather_holder = (0, 128, 255)
color_feather_holder_border = (0, 64, 128)
color_feather = (0, 128, 255)
color_feather_border = (0, 64, 128)


class Dummy_base():
    """
    Helper class for Dummy. Always use the Dummy-Class!
    """

    # baseline (d->dh)

    def __init__(self) -> None:
        """
        Constructor - create an empty dummy (or feather).
        Dummy consists of:
        figure = the polygone (Dummy-class: rectangle, Feather-Class: triangle)
        border_list = experimental list of distance zones. Till now, only used for drawing in cv2.
        color_figure, color_border: the colors for drawing in cv2
        """
        self.figure = geolib.Polygon([geolib.Vector(1, 1),
                                      geolib.Vector(1, 2), geolib.Vector(2, 2)])
        self.border_list = []
        self.color_figure = (0, 0, 0)
        self.color_border = (0, 0, 0)

    def set_zones(self, list_of_zones):
        """
        Define distance zones - Till now, only used for drawing in cv2; not used for statistical analysis.
        :param list_of_zones: list( tuple(zonename, bordersize)); zonename just for documentation
        :return: -
        """
        self.border_list = []
        for i, zone in enumerate(list_of_zones):
            self.border_list.append((i, zone[0], zone[1]))
        self.border_list.sort(key=lambda x: x[2])

    def plot_cv2(self, cv2_img, base_point):
        """
        Draw the Dummy/Feather to a cv2-image and move the dummy-drawing to the base point
        :param cv2_img: cv2-img object
        :param base_point: vector-object (moving vector)
        :return: -
        """

        def plot_figure(fig: geolib.Polygon, fig_color):
            nonlocal cv2_img
            for s in fig.straights:
                cv2.line(cv2_img, (int(s.base.x), int(s.base.y)), (int(s.base2.x), int(s.base2.y)), fig_color,
                         2, cv2.LINE_AA)

        plot_figure(self.figure.move(base_point), self.color_figure)
        for z in self.border_list:
            try:
                plot_figure(self.figure.get_border(z[2]).move(base_point), self.color_border)
            except Exception as e:
                pass
                #print("Cannot plot " + self.figure.__str__())


class Dummy(Dummy_base):
    """
    Class for a Dummy
    """

    def __init__(self, base_axis, aspect_ratio) -> None:
        """
        Constructor for a dummy. We will use the segment Broom4 to Broom3 (point D, DH)
        :param base_axis: straight-object of segment Broom4 to Broom3
        :param aspect_ratio: aspect ratio of the dummy
        """
        super().__init__()
        point_a = base_axis.base2.rotate90(base_axis.base, aspect_ratio / 2)
        point_d = base_axis.base2.rotate270(base_axis.base, aspect_ratio / 2)
        self.point_r = base_axis.base
        self.point_r2 = base_axis.base2
        self.axis = geolib.Straight(self.point_r, self.point_r2)
        self.aspect_ratio = aspect_ratio
        self.color_figure = color_feather_holder
        self.color_border = color_feather_holder_border
        self.figure = geolib.Polygon(
            [point_a, point_a.add(self.axis.direction), point_d.add(self.axis.direction), point_d])

    def rotate(self, angel_rad, center=SimpleVector(0.0, 0.0), aspect=1.0):
        self.__init__(self.axis.rotate(angel_rad, center, aspect), self.aspect_ratio)


class Feather(Dummy_base):
    """
    Class for the Feather
    """

    def __init__(self, point_dh, point_f1, point_f2) -> None:
        """
        Constructor for a feather. We will use the points Broom1, Broom2, and Broom3 (point F1, F2, DH)

        :param point_dh: vector-object of Broom3 (point DH)
        :param point_f1: vector-object of Broom1 (point F1)
        :param point_f2: vector-object of Broom2 (point F2)
        """

        super().__init__()
        self.point_dh = point_dh
        self.point_f1 = point_f1
        self.point_f2 = point_f2
        self.color_figure = color_feather
        self.color_border = color_feather_border
        self.figure = geolib.Polygon([point_dh, point_f1, point_f2])


class FeatherPattern(Feather):
    """
    Experimental class to construct alternative figures for the feather. Not used further; for analysis the default
    triangle type was used.
    """

    def __init__(self, dummy: Dummy, point_f1, point_f2, feather_type=0):
        """
        Constructor for different feather types to define feathers depending on the dummy and all feather points.
        :param dummy: Dummy-Object
        :param point_f1: vector-object of point F1
        :param point_f2: vector-object of point F1
        :param feather_type: 0 = default feather (triangle), 1= outline of a long feather
        """
        super().__init__(dummy.axis.base2, point_f1, point_f2)
        if feather_type == 1:
            projection_f = dummy.axis.orthogonal_projection(point_f1)
            vec_to_f = point_f1.sub(projection_f)
            self.figure = geolib.Polygon([
                point_f1,
                dummy.axis.base2.add(vec_to_f.s_mul(0.4)).add(dummy.axis.direction.s_mul(0.35)),
                dummy.axis.base2,
                dummy.axis.base2.add(vec_to_f.s_mul(-0.1)).add(dummy.axis.direction.s_mul(0.3)),
                dummy.axis.base2.add(vec_to_f.s_mul(0.3)).add(dummy.axis.direction.s_mul(1)),
                dummy.axis.base2.add(vec_to_f.s_mul(0.6)).add(dummy.axis.direction.s_mul(0.5)),
                point_f1.add(dummy.axis.direction.s_mul(0.2))])

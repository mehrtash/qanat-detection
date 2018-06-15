import time
import os
import numpy as np
import SimpleITK as sitk


def get_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S")


def create_project_tree(folders_list):
    for folder in folders_list:
        if not os.path.isdir(folder):
            os.mkdir(folder)


def get_list_from_file(filename):
    items = []
    with open(filename) as f:
        for item in f.read().splitlines():
            items.append(item)
    return items


def bounding_box(arr, padding=0, square=False):
    a = np.where(arr != 0)
    if a[0].size and a[1].size:
        min_ax0 = np.min(a[0])
        max_ax0 = np.max(a[0])
        min_ax1 = np.min(a[1])
        max_ax1 = np.max(a[1])
        if square:
            min_ax = min(min_ax0, min_ax1)
            max_ax = max(max_ax0, max_ax1)
            return min_ax - padding, max_ax + padding, min_ax - padding, max_ax + padding
        return min_ax0 - padding, max_ax0 + padding, min_ax1 - padding, max_ax1 + padding


def get_label_shape_stats(label, label_value=1, background_value=0):
    shape_stats_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_stats_filter.SetBackgroundValue(background_value)
    shape_stats_filter.ComputeFeretDiameterOn()
    shape_stats_filter.Execute(label)
    centroid = shape_stats_filter.GetCentroid(label_value)
    elongation = shape_stats_filter.GetElongation(label_value)
    spherical_radius = shape_stats_filter.GetEquivalentSphericalRadius(label_value)
    ellipsoid_diameter = shape_stats_filter.GetEquivalentEllipsoidDiameter(label_value)
    feret_diameter = shape_stats_filter.GetFeretDiameter(label_value)
    physical_size = shape_stats_filter.GetPhysicalSize(label_value)
    principal_moments = shape_stats_filter.GetPrincipalMoments(label_value)
    perimeter = shape_stats_filter.GetPerimeter(label_value)
    pixel_nos = shape_stats_filter.GetNumberOfPixels(label_value)
    principal_axes = shape_stats_filter.GetPrincipalAxes(label_value)
    roundness = shape_stats_filter.GetRoundness(label_value)
    flatness = shape_stats_filter.GetFlatness(label_value)
    bounding_box = shape_stats_filter.GetBoundingBox(label_value)
    return {'label_value': label_value,
            'centroid': centroid,
            'bounding_box': bounding_box,
            'elongation': elongation,
            'spherical_radius': spherical_radius,
            'feret_diameter': feret_diameter,
            'physical_size': physical_size,
            'principal_moments': principal_moments,
            'perimeter': perimeter,
            'pixel_nos': pixel_nos,
            'principal_axes': principal_axes,
            'roundness': roundness,
            'flatness': flatness,
            'ellipsoid_diameter': ellipsoid_diameter}


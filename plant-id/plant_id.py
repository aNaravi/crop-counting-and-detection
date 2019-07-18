import csv
from itertools import zip_longest
from collections import defaultdict, OrderedDict

import numpy as np
from sklearn.cluster import KMeans

from qgis.core import (
    QgsVectorLayer,
    QgsPointXY,
    QgsVectorFileWriter,
    QgsField
)

from PyQt5.QtCore import QVariant

points_shapefiles_dir = "/home/QGIS/Shapefiles/Window_1/Points/"
entries_shapefiles_dir = "/home/QGIS/Shapefiles/Window_1/Entries/"

week01_points = QgsVectorLayer(points_shapefiles_dir + "Week01/Week01.shp", "Week01_Points", "ogr")
week01_entries = QgsVectorLayer(entries_shapefiles_dir + "Week01/Week01.shp", "Week01_Entries", "ogr")
week02_points = QgsVectorLayer(points_shapefiles_dir + "Week02/Week02.shp", "Week02_Points", "ogr")
week02_entries = QgsVectorLayer(entries_shapefiles_dir + "Week02/Week02.shp", "Week02_Entries", "ogr")
week03_points = QgsVectorLayer(points_shapefiles_dir + "Week03/Week03.shp", "Week03_Points", "ogr")
week03_entries = QgsVectorLayer(entries_shapefiles_dir + "Week03/Week03.shp", "Week03_Entries", "ogr")
week04_points = QgsVectorLayer(points_shapefiles_dir + "Week04/Week04.shp", "Week04_Points", "ogr")
week04_entries = QgsVectorLayer(entries_shapefiles_dir + "Week04/Week04.shp", "Week04_Entries", "ogr")
week05_points = QgsVectorLayer(points_shapefiles_dir + "Week05/Week05.shp", "Week05_Points", "ogr")
week05_entries = QgsVectorLayer(entries_shapefiles_dir + "Week05/Week05.shp", "Week05_Entries", "ogr")
week06_points = QgsVectorLayer(points_shapefiles_dir + "Week06/Week06.shp", "Week06_Points", "ogr")
week06_entries = QgsVectorLayer(entries_shapefiles_dir + "Week06/Week06.shp", "Week06_Entries", "ogr")
week07_points = QgsVectorLayer(points_shapefiles_dir + "Week07/Week07.shp", "Week07_Points", "ogr")
week07_entries = QgsVectorLayer(entries_shapefiles_dir + "Week07/Week07.shp", "Week07_Entries", "ogr")


def add_attribute(points_layer, name, type):
    points_layer.dataProvider().addAttributes([QgsField(name, type)])
    points_layer.updateFields()


def set_plant_ID(points_layer):
    add_attribute(points_layer, "PID", QVariant.String)
    field_idx = points_layer.fields().indexFromName("PID")

    entries = list(range(101,167)) + list(range(201,267)) + list(range(301,367))
    points_layer.startEditing()
    for entry in entries:
        points = np.array([[F.geometry().asPoint().x(), F.geometry().asPoint().y(), F.id()]
                           for F in points_layer.getFeatures() if F.attribute('Entry') == entry])
        y_coordinates = points[:,1].reshape(-1,1)
        clusters = KMeans(n_clusters=2).fit(y_coordinates)
        labels = clusters.labels_

        if points[labels == 0, 1].mean() < points[labels == 1, 1].mean():
            lower, upper = 0, 1
        else:
            lower, upper = 1, 0

        for i, p in enumerate(np.vstack((points[labels == upper][np.argsort(points[labels == upper, 0])],
                                         points[labels == lower][np.argsort(points[labels == lower, 0])])), 1):
            points_layer.changeAttributeValue(p[2], field_idx,
                                              points_layer.name().replace('eek', '') + '-E' + str(entry) + '-P' + '{:02}'.format(i))
    points_layer.commitChanges()


def set_entry_numbers(points_layer, entries_layer, entry_field='Entry_no'):
    add_attribute(points_layer, "Entry", QVariant.Int)
    field_idx = points_layer.fields().indexFromName("Entry")

    points_layer.startEditing()
    for E in entries_layer.getFeatures():
        entry = E.attribute(entry_field)
        for P in points_layer.getFeatures():
            if P.geometry().intersects(E.geometry()):
                points_layer.changeAttributeValue(P.id(), field_idx, entry)
    points_layer.commitChanges()


def rotate_points_layer(points_layer, angle, point=None):
    if point is None:
        coordinates = []
        for F in points_layer.getFeatures():
            coordinates.append((F.geometry().asPoint().x(), F.geometry().asPoint().y()))
        point = np.array(coordinates).mean(0)

    points_layer.startEditing()
    for i,F in enumerate(points_layer.getFeatures()):
        G = F.geometry()
        G.rotate(angle, QgsPointXY(*point))
        points_layer.changeGeometry(fid=i, geometry=G)
    points_layer.commitChanges()

    return point


def nearest_neighbours(reference_points_layer, points_layer):
    def ellipses(tPoint, angle, semi_major_axis, semi_minor_axis):
        def neighborhood(P):
            return (A * (P[0] ** 2)) + (B * P[0] * P[1]) + (C * (P[1] ** 2)) + (D * P[0]) + (E * P[1]) + F <= 0

        Sin, Cos = np.sin(angle), np.cos(angle)
        centers = tPoint + np.array([Cos, Sin]) * semi_major_axis * np.array([[0.8],[-0.8]])
        A = (semi_major_axis ** 2) * (Sin ** 2) + (semi_minor_axis ** 2) * (Cos ** 2)
        B = 2 * ((semi_minor_axis ** 2) - (semi_major_axis ** 2)) * Sin * Cos
        C = (semi_major_axis ** 2) * (Cos ** 2) + (semi_minor_axis ** 2) * (Sin ** 2)
        D = -(2 * A * centers[:,0]) - (B * centers[:,1])
        E = -(B * centers[:,0]) - (2 * C * centers[:,1])
        F = (A * (centers[:,0] ** 2)) + (B * centers[:,0] * centers[:,1]) + (C * (centers[:,1] ** 2)) \
            - ((semi_major_axis ** 2) * (semi_minor_axis ** 2))

        return neighborhood

    def process_entry(reference_points, reference_points_ids, points, points_ids):
        least_squares_fit = np.polynomial.polynomial.Polynomial.fit

        y_coordinates = reference_points[:,1].reshape(-1,1)
        clusters = KMeans(n_clusters=2).fit(y_coordinates)
        labels = clusters.labels_

        rows, ids, intercepts, slopes = {}, {}, {}, {}

        for k in range(2):
            rows[k] = reference_points[labels == k]
            ids[k] = reference_points_ids[labels == k]
            intercepts[k], slopes[k] = least_squares_fit(rows[k][:,0], rows[k][:,1], 1).coef

        semi_major_axis = abs(intercepts[1] - intercepts[0]) / 3
        semi_minor_axis = semi_major_axis * (2 / 3)

        for k in range(2):
            angle = np.arctan(-1 / slopes[k])
            for ID, point in zip(ids[k], rows[k]):
                neighborhood = ellipses(point, angle, semi_major_axis, semi_minor_axis)
                valid = np.apply_along_axis(neighborhood, axis=1, arr=points).any(axis=1)
                if np.any(valid):
                    sq_distances = ((points[valid] - point) ** 2).sum(axis=1)
                    reference_points_NN[ID].update(dict(zip(points_ids[valid], 1 / sq_distances)))
                    for eID, rID, d in zip_longest(points_ids[valid], [ID], 1 / sq_distances, fillvalue=ID):
                        points_NN[eID].update({rID: d})

    entries = list(range(101,167)) + list(range(201,267)) + list(range(301,367))
    reference_points_NN, points_NN = defaultdict(dict), defaultdict(dict)

    for entry in entries:
        reference_points = np.array([F.geometry().asPoint() for F in reference_points_layer.getFeatures() if F.attribute('Entry') == entry])
        reference_points_ids = np.array([F.attribute('PID') for F in reference_points_layer.getFeatures() if F.attribute('Entry') == entry])
        points = np.array([F.geometry().asPoint() for F in points_layer.getFeatures() if F.attribute('Entry') == entry])
        points_ids = np.array([F.attribute('PID') for F in points_layer.getFeatures() if F.attribute('Entry') == entry])
        process_entry(reference_points, reference_points_ids, points, points_ids)

    return reference_points_NN, points_NN


def spatial_matching(reference_points_data, points_data):
    matches = OrderedDict()
    while points_data.keys():
        start_key = list(points_data.keys())[0]
        chain = [start_key]
        while chain:
            x = chain[-1]
            if x[:2] == start_key[:2]:
                y = max(points_data.get(x, {}), key=points_data.get(x, {}).get, default='-')
                if len(chain) > 1 and y == chain[-2]:
                    del chain[-2:]
                    del points_data[x]
                    del reference_points_data[y]
                    matches[x] = y
                else:
                    if y == '-':
                        if matches.get(x):  # if points_data.get(x) == None
                            del reference_points_data[chain[-2]][x]
                            del chain[-1]
                        else:  # if points_data.get(x) == {}
                            del chain[-1]
                            del points_data[x]
                    else:
                        chain.append(y)
            else:
                y = max(reference_points_data.get(x, {}), key=reference_points_data.get(x, {}).get, default='-')
                if len(chain) > 1 and y == chain[-2]:
                    del chain[-2:]
                    del points_data[y]
                    del reference_points_data[x]
                    matches[y] = x
                else:
                    if y == '-':
                        del points_data[chain[-2]][x]
                        del chain[-1]
                    else:
                        chain.append(y)

    return matches, OrderedDict(zip(matches.values(), matches.keys()))


def detection_errors(reference_points_layer, points_layer, reference_points_data, points_data, matches):
    # False Negatives
    isolated_in_reference_points_layer = set.difference(
        set([f.attribute('PID') for f in reference_points_layer.getFeatures()]),
        set(reference_points_data.keys())
    )
    unmatched_in_reference_points_layer = set.difference(set(reference_points_data.keys()), set(matches.values()))
    # False Positives
    isolated_in_points_layer = set.difference(set([f.attribute('PID') for f in points_layer.getFeatures()]), set(points_data.keys()))
    unmatched_in_points_layer = set.difference(set(points_data.keys()), set(matches.keys()))

    # Select False Negatives
    reference_points_dict = {F.attribute('PID'):F.id() for F in reference_points_layer.getFeatures()}
    reference_points_layer.selectByIds(
        [reference_points_dict[PID] for PID in set.union(isolated_in_reference_points_layer, unmatched_in_reference_points_layer)]
    )
    # Select False Positives
    points_dict = {F.attribute('PID'):F.id() for F in points_layer.getFeatures()}
    points_layer.selectByIds([points_dict[PID] for PID in set.union(isolated_in_points_layer, unmatched_in_points_layer)])


def create_matches_csv(dictionary, filename='matches', folder='/home/QGIS/Extras/'):
    with open(folder + filename + '.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(dictionary.items())


def run(reference_points_layer, reference_entries_layer, points_layer, entries_layer,
        selected_features_filename='/tmp/selected_features.shp'):
    if reference_points_layer.fields().indexFromName("Entry") == -1:
        set_entry_numbers(reference_points_layer, reference_entries_layer)
    centroid1 = rotate_points_layer(reference_points_layer, -3.5)
    if reference_points_layer.fields().indexFromName("PID") == -1:
        set_plant_ID(reference_points_layer)

    if points_layer.fields().indexFromName("Entry") == -1:
        set_entry_numbers(points_layer, entries_layer, "ENTRY")
    centroid2 = rotate_points_layer(points_layer, -3.5)
    if points_layer.fields().indexFromName("PID") == -1:
        set_plant_ID(points_layer)

    reference_points_data, points_data = nearest_neighbours(reference_points_layer, points_layer)
    matches, _ = spatial_matching(reference_points_data.copy(), points_data.copy())

    rotate_points_layer(reference_points_layer, 3.5, centroid1)
    rotate_points_layer(points_layer, 3.5, centroid2)

    points_layer_dict = {F.attribute('PID'):F.id() for F in points_layer.getFeatures()}
    points_layer.selectByIds([points_layer_dict[PID] for PID in matches.keys()])
    # isolated_in_points_layer = set.difference(set([f.attribute('PID') for f in points_layer.getFeatures()]), set(points_data.keys()))
    # unmatched_in_points_layer = set.difference(set(points_data.keys()), set(matches.keys()))
    # points_layer.selectByIds([points_layer_dict[PID] for PID in set.union(isolated_in_points_layer, unmatched_in_points_layer)])

    QgsVectorFileWriter.writeAsVectorFormat(points_layer, fileName=selected_features_filename,
                                            fileEncoding='System', driverName='ESRI Shapefile', onlySelected=True)


def create_binary_csvs(points_layer, csv_dir='/home/QGIS/Extras/'):
    def write_coordinates_to_csv(coordinates, filename):
        with open(csv_dir + filename + '.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(coordinates)

    entries = list(range(101,167)) + list(range(201,267)) + list(range(301,367))
    for entry in entries:
        points_layer.selectByExpression("Entry = " + str(entry))
        coordinates = []
        for P in points_layer.getSelectedFeatures():
            coordinates.append((P.geometry().asPoint().x(), P.geometry().asPoint().y()))
        points_layer.selectByIds([])
        write_coordinates_to_csv(coordinates, 'binary_' + str(entry))

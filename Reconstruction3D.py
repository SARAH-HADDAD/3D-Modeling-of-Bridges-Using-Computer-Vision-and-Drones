from h5_to_db import add_keypoints, add_matches, COLMAPDatabase
import pycolmap
import os



def import_into_colmap(img_dir,
                       feature_dir ='.featureout',
                       database_path = 'colmap.db',
                       img_ext='.jpg'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, 'simple-radial', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )

    db.commit()
    return


feature_dir = 'featureout_loftr'
dirname = 'dirname'
output_path = 'colmap_rec_loftr'

database_path = 'colmap_loftr.db'

if os.path.exists(database_path):
    os.remove(database_path)


import_into_colmap(dirname,feature_dir=feature_dir, database_path=database_path)


pycolmap.match_exhaustive(database_path)

if not os.path.isdir(output_path):
    os.makedirs(output_path)

maps = pycolmap.incremental_mapping(database_path, dirname, output_path)
maps[0].write(output_path)
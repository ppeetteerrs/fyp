from glob import glob

import pandas as pd
import pydicom as pyd
from tqdm.contrib.concurrent import process_map
from utils.config import COVID_19_DIR


def parse_record(dcm_path: str):
    # Read Dicom data
    data = pyd.dcmread(dcm_path)

    # Get subject and image ID
    subject_id = dcm_path.split("/")[-2].replace("subject_", "")
    image_id = dcm_path.split("/")[-1].replace(".dcm", "")

    # Return data dictionary
    return {
        "Subject": subject_id,
        "ID": image_id,
        "Path": dcm_path,
        "PhotometricInterpretation": data.PhotometricInterpretation,
        "SliceLocation": float(data.SliceLocation),
        "SliceThickness": float(data.SliceThickness),
        "ImagePositionPatient0": data.ImagePositionPatient[0],
        "ImagePositionPatient1": data.ImagePositionPatient[1],
        "ImagePositionPatient2": data.ImagePositionPatient[2],
        "PatientPosition": data.PatientPosition,
        "ImageOrientationPatient": data.ImageOrientationPatient,
        "SeriesDescription": data.SeriesDescription.lower(),
        "SeriesNumber": data.SeriesNumber,
        "PixelSpacing0": data.PixelSpacing[0],
        "PixelSpacing1": data.PixelSpacing[1],
        "ImageType0": data.ImageType[0],
        "ImageType1": data.ImageType[1],
        "ImageType2": data.ImageType[2],
        "ImageType3": data.ImageType[3] if len(data.ImageType) > 3 else "NONE",
        "HighBit": data.HighBit,
    }


if __name__ == "__main__":
    dcms = list(glob(f"{COVID_19_DIR}/**/**.dcm"))
    records = process_map(parse_record, dcms, chunksize=1)
    df = pd.DataFrame.from_records(records)
    df.to_csv(COVID_19_DIR / "meta.csv", index=False)

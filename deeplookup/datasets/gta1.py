from io import BytesIO
from pathlib import Path
from typing import List
from zipfile import ZipFile

import pandas as pd
import tensorflow_datasets as tfds
from six.moves import urllib


_DESCRIPTION = (
    "The dataset is a collection of Domain Generation Algorithms (DGA), "
    "Domain Tunneling Algorithms (DTA), and legitimate DNS name used to "
    "access popular Internet resources, safe domains hosted by DNS "
    "providers as well as domains of Content Delivery Network (CDN). "
    "It contains 1.65M labeled domain names divided into 55 classes, "
    "4 of which are DTA, 50 of which are DGA and 1 legitimate class."
)

_GTA_URL = "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com"
_GTA_DATA_FILENAME = "2wzf9bz7xr-1.zip"

_NAMES = [
    "legit",
    "qadars",
    "kraken_v2",
    "chinad",
    "gozi_gpl",
    "murofet_v3",
    "murofet_v1",
    "bedep",
    "kraken_v1",
    "vawtrak_v1",
    "banjori",
    "tuns",
    "murofet_v2",
    "fobber_v1",
    "cryptolocker",
    "pykspa_noise",
    "ramnit",
    "suppobox_3",
    "suppobox_1",
    "simda",
    "nymaim",
    "ramdo",
    "suppobox_2",
    "qakbot",
    "sisron",
    "tempedreve",
    "gozi_nasa",
    "tinba",
    "ranbyus_v1",
    "padcrypt",
    "symmi",
    "necurs",
    "matsnu",
    "dnscapy",
    "shiotob",
    "iodine",
    "gozi_rfc4343",
    "dns2tcp",
    "vawtrak_v2",
    "proslikefan",
    "fobber_v2",
    "pizd",
    "dyre",
    "zeus-newgoz",
    "alureon",
    "gozi_luther",
    "locky",
    "rovnix",
    "pykspa",
    "dircrypt",
    "pushdo",
    "vawtrak_v3",
    "corebot",
    "ccleaner",
    "ranbyus_v2",
]

GTA_NUM_CLASSES = len(_NAMES)


class Gta1(tfds.core.GeneratorBasedBuilder):
    """DGTA-BENCH dataset."""

    URL = _GTA_URL
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "domain": tfds.features.Text(),
                    "class": tfds.features.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("domain", "class"),
            homepage="https://data.mendeley.com/datasets/2wzf9bz7xr/1",
        )

    def _split_generators(
        self, dl_manager: tfds.download.DownloadManager
    ) -> List[tfds.core.SplitGenerator]:
        """Define the train and test split."""
        dataset_files = dl_manager.download(
            {
                "dataset_archive": urllib.parse.urljoin(self.URL, _GTA_DATA_FILENAME),
            }
        )

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(dataset_archive_path=dataset_files["dataset_archive"]),
            ),
        ]

    def _generate_examples(
        self, dataset_archive_path: str
    ) -> tfds.core.split_builder.SplitGenerator:
        """Generate examples of domain names with associated class label."""
        with ZipFile(str(dataset_archive_path)) as dataset_archive:
            root_dirname = Path(_GTA_DATA_FILENAME).stem
            parquet_path = f"/{root_dirname}/gta-v1.parquet"

            with dataset_archive.open(parquet_path) as parquet_file:
                parquet_bytes = BytesIO(parquet_file.read())
                df = pd.read_parquet(parquet_bytes)

        numpy_dataset = df.to_numpy()
        for row_number, row in enumerate(numpy_dataset):
            yield row_number, {"domain": row[0], "class": row[1]}

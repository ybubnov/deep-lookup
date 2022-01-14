# Deep Lookup - Deep Learning for Domain Name System

## Installation

### Installation Using PyPi

```sh
pip install deeplookup
```

## Using DeepLookup

DeepLookup provides a `Resolver` instance that inherits [`dns.resolver.Resolver`](dns-resolver)
```py
from deeplookup import Resolver


resolver = Resolver()

for ip in resolver.resolve("google.com", "A"):
    print(f"ip: {ip.to_text()}")
```

The code above performs a verification of a queried name using a neural network trained
to detect malicious queries ([DGAs][dga-wiki] and tunnels). For the example above the
output will look like following:
```sh
ip: 142.250.184.206
```

When the queried name is generated using domain generation algorithm, the resolver throws
[`dns.resolver.NXDOMAIN`](dns-nxdomain) without even accessing a remote name server.
```py
for ip in resolver.resolve("mjewnjixnjaa.com", "A"):
    print(f"ip: {ip.to_text()}")
```

The example above throws [`dns.resolver.NXDOMAIN`](dns-nxdomain) error with the following
message:
```sh
dns.resolver.NXDOMAIN: The DNS query name does not exist: mjewnjixnjaa.com.
```

## Training

The model is trained using [tfx](txf) pipeline, where the training dataset is uploaded,
split into the training and evaluation subsets and then used to fit the neural network.

In order to trigger the training pipeline use the following command:
```sh
python -m deeplookup.pipeline.gta1
```

This command creates a folder called "tfx", where all artifacts are persisted. See the
`tfx/pipelines/gta1/serving_model/gta1/*` folder to access the model in HDF5 format.

## Publications
1. Bubnov Y., Ivanov N. (2020) Text analysis of DNS queries for data exfiltration protection of computer networks, [_Informatics_][Informatics, 2020], 3, 78-86.
2. Bubnov Y., Ivanov N. (2020) Hidden Markov model for malicious hosts detection in a computer network, [_Journal of BSU. Mathematics and Informatics_][BSU, 2020], 3, 73-79.
3. Bubnov Y., Ivanov N. (2021) DGA domain detection and botnet prevention using Q-learning for POMDP, [_Doklady BGUIR_][BGUIR, 2021], 2, 91-99.

## Datasets

The most robust dataset [DGTA-BENCH][DGTA, 2021] is available through
[tensorflow datasets](https://www.tensorflow.org/datasets) API and used for training
other neural network architectures:
```py
import deeplookup.datasets as dlds
import tensorflow_datasets as tfds

ds = tfds.load("gta1", shuffle_files=True)

for example in ds.take(1):
  domain, label = example["domain"], example["class"]
```

1. Bubnov Y. (2019) DNS Tunneling Queries for Binary Classification, [_Mendeley Data_][DTQBC, 2019], v1.
2. Zago M., Perez. M.G., Perez G.M. (2020) UMUDGA - University of Murcia Domain Generation Algorithm Dataset, [_Mendeley Data_][UMUDGA, 2020], v1.
3. Bybnov Y. (2021) DGTA-BENCH - Domain Generation and Tunneling Algorithms for Benchmark, [_Mendeley Data_][DGTA, 2021], v1.


[Informatics, 2020]: https://doi.org/10.37661/1816-0301-2020-17-3-78-86
[BSU, 2020]: https://doi.org/10.33581/2520-6508-2020-3-73-79
[BGUIR, 2021]: https://doi.org/10.35596/1729-7648-2021-19-2-91-99
[UMUDGA, 2020]: http://dx.doi.org/10.17632/y8ph45msv8.1
[DTQBC, 2019]: http://dx.doi.org/10.17632/mzn9hvdcxg.1
[DGTA, 2021]: http://dx.doi.org/10.17632/2wzf9bz7xr.1

[dga-wiki]: https://en.wikipedia.org/wiki/Domain_generation_algorithm
[dns-resolver]: https://dnspython.readthedocs.io/en/latest/resolver-class.html
[dns-nxdomain]: https://dnspython.readthedocs.io/en/latest/exceptions.html#dns.resolver.NXDOMAIN
[tfx]: https://www.tensorflow.org/tfx

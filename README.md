# Deep Lookup - Deep Learning for Domain Name System

## Installation

### Installation Using PyPi

```sh
```

## Using DeepLookup

DeepLookup provides a `Resolver` instance that inhertits [`dns.resolver.Resolver`](dns-resolver)
```py
from deeplookup import Resolver


resolver = Resolver()

for ip in resolver.resolve("google.com", "A"):
    print("ip: ", ip.to_text())
```

The code above, performs a veficiation of a queried name using a nerual network trained
to detect malicious queries ([DGAs][dga-wiki] and tunnels). For the example above the
output will look like following:
```sh
ip:  142.250.184.206
```

When the queried name is generated using domain generation algorith, the resolver throws
[`dns.resolver.NXDOMAIN`](dns-nxdomain) without even accessing a remote nameserver.
```
for ip in resolver.resolve("mjewnjixnjaa.com", "A"):
    print("ip: ", ip.to_text())
```

The output of the example above will throw the following error:
```sh
dns.resolver.NXDOMAIN: The DNS query name does not exist: mjewnjixnjaa.com.
```

## Publications
1. Bubnov Y., Ivanov N. (2020) Text analysis of DNS queries for data exfiltration protection of computer networks, [_Informatics_][Informatics, 2020], 3, 78-86.
2. Bubnov Y., Ivanov N. (2020) Hidden Markov model for malicious hosts detection in a computer network, [_Journal of BSU. Mathematics and Informatics_][BSU, 2020], 3, 73-79.
3. Bubnov Y., Ivanov N. (2021) DGA domain detection and botnet prevention using Q-learning for POMDP, [_Doklady BGUIR_][BGUIR, 2021], 2, 91-99.

## Datasets
1. Bubnov Y. (2019) DNS Tunneling Queries for Binary Classification, [_Mendeley Data_][DTQBC, 2019], v1.
2. Zago M., Perez. M.G., Perez G.M. (2020) UMUDGA - University of Murcia Domain Generation Algorithm Dataset, [_Mendeley Data_][UMUDGA, 2020], v1.


[Informatics, 2020]: https://doi.org/10.37661/1816-0301-2020-17-3-78-86
[BSU, 2020]: https://doi.org/10.33581/2520-6508-2020-3-73-79
[BGUIR, 2021]: https://doi.org/10.35596/1729-7648-2021-19-2-91-99
[UMUDGA, 2020]: http://dx.doi.org/10.17632/y8ph45msv8.1
[DTQBC, 2019]: http://dx.doi.org/10.17632/mzn9hvdcxg.1

[dga-wiki]: https://en.wikipedia.org/wiki/Domain_generation_algorithm
[dns-resolver]: https://dnspython.readthedocs.io/en/latest/resolver-class.html
[dns-nxdomain]: https://dnspython.readthedocs.io/en/latest/exceptions.html#dns.resolver.NXDOMAIN

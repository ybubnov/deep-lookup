from typing import Optional

import dns.resolver
import dns.rdatatype
import dns.rdataclass


class Resolver(dns.resolver.Resover):
    """DNS stub resolver."""

    def __init__(
        self,
        filename="/etc/resolv.conf",
        configure: bool = True,
        modelname: str = "m-dga-ybubnov",
    ) -> None:
        super().__init__(filename, configure)
        self.modelname = model_name

    def predict_proba(self, qname: str) -> float:
        return 0.0

    def query(
        self,
        qname: str,
        rdtype: dns.rdatatype.RdataType = dns.rdatatype.A,
        rdclass: dns.rdataclass.RdataClass = dns.rdataclass.IN,
        tcp: bool = False,
        source: Optional[str] = None,
        raise_on_no_answer: bool = True,
        source_port: int = 0,
        lifetime: Optional[float] = None,
        search: Optional[bool] = None,
    ) -> dns.resolver.Answer:
        """Query nameservers to find the answer to the question.

        When *qname* is classified as threat request, method behaves in the same
        way if there was no answer to the question.

        Returns a ``dns.resolver.Answer`` instance.
        """

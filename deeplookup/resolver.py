from pathlib import Path
from typing import Optional

import dns
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.resolver
import tensorflow as tf

from deeplookup import weights
from deeplookup.pipelines.gta1_trainer import _tokenize_domain


class Resolver(dns.resolver.Resolver):
    """DNS stub resolver."""

    def __init__(
        self,
        filename="/etc/resolv.conf",
        configure: bool = True,
        modelname: str = "gta-v1",
    ) -> None:
        super().__init__(filename, configure)

        self.model_path = Path(weights.__file__).parent / (modelname + ".h5")
        self.model = tf.keras.models.load_model(self.model_path)

    def predict_proba(self, qname: str) -> float:
        """Returns a probability between 0 and 1 that given *qname* is malicious."""
        x = _tokenize_domain(tf.constant([[qname]]))
        return self.model.predict(x.numpy())[0][1]

    def resolve(
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
        min_proba: float = 0.5,
    ) -> dns.resolver.Answer:
        """Query nameservers to find the answer to the question.

        When *qname* is classified as threat request, method behaves in the same
        way if there was no answer to the question.

        Returns a ``dns.resolver.Answer`` instance.
        """
        proba = self.predict_proba(qname)
        if proba > min_proba:
            raise dns.resolver.NXDOMAIN(qnames=[dns.name.from_text(qname)])

        return super().resolve(
            qname=qname,
            rdtype=rdtype,
            rdclass=rdclass,
            tcp=tcp,
            source=source,
            raise_on_no_answer=raise_on_no_answer,
            source_port=source_port,
            lifetime=lifetime,
            search=search,
        )

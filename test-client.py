import base64
import logging

from Cryptodome.PublicKey import ECC
from Cryptodome.Hash import SHA256
from Cryptodome.Signature import DSS
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

import ndn.utils
from ndn.app import NDNApp
from ndn.types import InterestNack, InterestTimeout, InterestCanceled, ValidationFailure
from ndn.encoding import Name, Component, InterestParam, parse_and_check_tl, FormalName, SignatureType, SignaturePtrs
from ndn.app_support.security_v2 import SafeBag, SecurityV2TypeNumber, parse_certificate, CertificateV2Value

from security_helper import *

logging.basicConfig(format='[{asctime}]{levelname}:{message}',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    style='{')


app = NDNApp()
local_anchor = None
testbed_signed = None
key_bag = b''
client_prv_key = b''

# client's packets signer
client_identity = None

async def main():
    global local_anchor

    import_safebag("sec/client.safebag", "1234")

    # parse again to read prv key into memory 
    request = CertRequest()
    with open("sec/client.safebag", "r") as safebag:
        wire = safebag.read()
        wire = base64.b64decode(wire)
        wire = parse_and_check_tl(wire, SecurityV2TypeNumber.SAFE_BAG)
        bag = SafeBag.parse(wire)

        # attach the testbed-signed certificate
        request.testbed_signed = bag.certificate_v2

        # parse the key bag to obtain private key
        testbed_signed = CertificateV2Value.parse(bag.certificate_v2)
        key_bag = bytes(bag.encrypted_key_bag)
        privateKey = serialization.load_der_private_key(key_bag, password=b'1234', backend=default_backend())
        client_prv_key = privateKey.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())

    # parse trust anchor and self-assigns a name, then create self-signed certificate
    with open("sec/client.anchor", "r") as anchor:
        wire = anchor.read()
        wire = base64.b64decode(wire)
        local_anchor = parse_certificate(wire)

        # self-assign a name and create corresponding key pair
        client_name = local_anchor.name[:-4] + [testbed_signed.name[-5]]
        client_identity = app.keychain.touch_identity(client_name)

        # attach newly generated self-assigned certificate
        cert = client_identity.default_key().default_cert().data
        cert = parse_certificate(cert)
        request.self_signed = cert.encode()

    try:
        # express the first interest to fetch a token/secret code
        timestamp = ndn.utils.timestamp()
        name = Name.from_str('/edge/_ca/new-cert') + [Component.from_timestamp(timestamp)]
        logging.info(f'Sending Interest {Name.to_str(name)}, {InterestParam(must_be_fresh=True, lifetime=6000)}')
        data_name, meta_info, content = await app.express_interest(
            name, app_param=request.encode(), must_be_fresh=True, can_be_prefix=False, lifetime=6000,
            identity=client_identity, validator=verify_ecdsa_signature)
        
        # sign it use the private key, to prove the certificate possesion
        h = SHA256.new()
        h.update(bytes(content))
        pk = ECC.import_key(client_prv_key)
        signature = DSS.new(pk, 'fips-186-3', 'der').sign(h)
        logging.info(f'Getting Data {Name.to_str(name)}, begin signing the token {bytes(content)}')

        # express the second interest to fetch the issued certificate
        name = Name.from_str('/edge/_ca/challenge') + [Component.from_timestamp(timestamp)]
        logging.info(f'Sending Interest {Name.to_str(name)}, {InterestParam(must_be_fresh=True, lifetime=6000)}')
        data_name, meta_info, content = await app.express_interest(
            name, app_param=signature, must_be_fresh=True, can_be_prefix=False, lifetime=6000, 
            identity=client_identity, validator=verify_ecdsa_signature)

        # parse the issued certificate and install
        issued_cert = parse_certificate(content)
        logging.info("Issued certificate: %s", Name.to_str(issued_cert.name))
        app.keychain.import_cert(Name.to_bytes(issued_cert.name[:-2]), Name.to_bytes(issued_cert.name), bytes(content))

    except InterestNack as e:
        print(f'Nacked with reason={e.reason}')
    except InterestTimeout:
        print(f'Timeout')
    except InterestCanceled:
        print(f'Canceled')
    except ValidationFailure:
        print(f'Data failed to validate')
        app.shutdown()


async def verify_ecdsa_signature(name: FormalName, sig: SignaturePtrs) -> bool:
    global local_anchor

    sig_info = sig.signature_info
    covered_part = sig.signature_covered_part
    sig_value = sig.signature_value_buf
    if not sig_info or sig_info.signature_type != SignatureType.SHA256_WITH_ECDSA:
        return False
    if not covered_part or not sig_value:
        return False
    key_name = sig_info.key_locator.name[0:]
    logging.debug('Extract key_name: %s', Name.to_str(key_name))

    # local trust anchor pre-shared, already know server's public key
    key_bits = None
    try:
        key_bits = bytes(local_anchor.content)
    except (KeyError, AttributeError):
        logging.debug('Cannot load pub key from received certificate')
        return False
    pk = ECC.import_key(key_bits)
    verifier = DSS.new(pk, 'fips-186-3', 'der')
    sha256_hash = SHA256.new()
    for blk in covered_part:
        sha256_hash.update(blk)
    try:
        verifier.verify(sha256_hash, bytes(sig_value))
    except ValueError:
        return False
    return True
        
if __name__ == '__main__':
    # clear the environment just in case
    clear_client_keychain()
    app.run_forever(after_start=main())
    clear_client_keychain()
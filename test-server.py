import base64
import logging
from typing import Optional
from os import urandom
import time
from datetime import date, timedelta
import asyncio

from Cryptodome.PublicKey import ECC
from Cryptodome.Hash import SHA256
from Cryptodome.Signature import DSS
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

from ndn.app_support.security_v2 import SafeBag, SecurityV2TypeNumber, parse_certificate, CertificateV2Value, derive_cert
from ndn.security.signer import Sha256WithEcdsaSigner
from ndn.encoding import Name, InterestParam, BinaryStr, FormalName, parse_and_check_tl, SignatureType, SignaturePtrs
from sec.bootApp import NDNApp

from security_helper import *

logging.basicConfig(format='[{asctime}]{levelname}:{message}', datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG, style='{')

app = NDNApp()
# security components
trust_anchor = None
client_self_signed = None
testbed_signed = None
token = b''

# server's packets signer
signer = None


def on_cert_validation_success(param):
    global signer, token
    token = urandom(16)
    logging.info("Certifiate validation succeed")  
    app.put_data(param, content=token, freshness_period=10000, signer=signer)

def on_cert_validation_failure(param):
    global signer
    content = "validation failed".encode()
    app.put_data(param, content=content, freshness_period=10000, signer=signer)

def on_interest_verify_success(param):
    global testbed_signed
    logging.info("Interest verification succeed") 

    # validate client's testbed-signed certificate
    cert_sig_ptrs = SignaturePtrs(
        signature_info=testbed_signed.signature_info,
        signature_covered_part=testbed_signed._sig_cover_part.get_arg(param[0]),
        signature_value_buf=testbed_signed.signature_value,
    )
    asyncio.create_task(_validator_wrapper(testbed_signed.name, cert_sig_ptrs, verify_ecdsa_signature,
                                           on_cert_validation_success, on_cert_validation_failure, param[1]))

def on_interest_verify_failure(param):
    return

# wrapper to run async validator within sync on_interest
async def _validator_wrapper(name: FormalName, sig_ptrs: SignaturePtrs, _validator, success_cb, failure_cb, param):
    validation_result = await _validator(name, sig_ptrs)
    if validation_result is True:
        success_cb(param)
    else:
        failure_cb(param)

# if public key is already at local
async def verify_known_ecdsa_signature(name: FormalName, sig: SignaturePtrs) -> bool:
    global client_self_signed
    sig_info = sig.signature_info
    covered_part = sig.signature_covered_part
    sig_value = sig.signature_value_buf
    if not sig_info:
        # TODO: Android app with jNDN cannot sign the interest yet
        return True
    if not sig_info or sig_info.signature_type != SignatureType.SHA256_WITH_ECDSA:
        return False
    if not covered_part or not sig_value:
        return False
    try:
        key_bits = bytes(client_self_signed.content)
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

# recusive verifier til reach the testbed root
# Tianyuan: currently I don't check validity
async def verify_ecdsa_signature(name: FormalName, sig: SignaturePtrs) -> bool:
    global trust_anchor

    logging.debug("Validating certificate {}".format(Name.to_str(name)))
    sig_info = sig.signature_info
    covered_part = sig.signature_covered_part
    sig_value = sig.signature_value_buf
    if not sig_info or sig_info.signature_type != SignatureType.SHA256_WITH_ECDSA:
        return False
    if not covered_part or not sig_value:
        return False
    key_name = sig_info.key_locator.name[0:]
    logging.debug('Extracting key_name: {}'.format(Name.to_str(key_name)))

    # check it's testbed root
    key_bits = None
    if Name.to_str(key_name) == Name.to_str(trust_anchor.name[:-2]):
        logging.debug('Reaching the trust anchor, begin to return')
        key_bits = bytes(trust_anchor.content)
    else:
        try:
            cert_name, meta_info, content, raw_packet = await app.express_interest(key_name, must_be_fresh=True, can_be_prefix=True, 
                                                        lifetime=6000, need_raw_packet=True, validator=verify_ecdsa_signature)
            # certificate itself is a Data packet
            cert = parse_certificate(raw_packet)
            # load public key from the data content
            key_bits = None
            try:
                key_bits = bytes(content)
            except (KeyError, AttributeError):
                logging.debug('Cannot load pub key from received certificate')
                return False
        except InterestNack as e:
            logging.debug(f'Nacked with reason={e.reason}')
        except InterestTimeout:
            logging.debug(f'Timeout')
        except InterestCanceled:
            logging.debug(f'Canceled')
        except ValidationFailure:
            logging.debug(f'Data failed to validate')
    pk = ECC.import_key(key_bits)
    verifier = DSS.new(pk, 'fips-186-3', 'der')
    sha256_hash = SHA256.new()
    for blk in covered_part:
        sha256_hash.update(blk)
    try:
        verifier.verify(sha256_hash, bytes(sig_value))
    except ValueError:
        logging.debug('Certificate validation failed! %s', Name.to_str(name))
        return False
    logging.debug('Certificate validated! %s', Name.to_str(name))
    return True

@app.route('/edge/_ca/new-cert', need_sig_ptrs=True)
def on_new_cert_interest(name: FormalName, param: InterestParam, _app_param: Optional[BinaryStr], sig_ptrs: SignaturePtrs):
    global client_self_signed, testbed_signed
    logging.info("New-cert Interest")
    request = CertRequest.parse(_app_param)
    
    markers = {}
    client_self_signed = CertificateV2Value.parse(request.self_signed)
    testbed_signed = CertificateV2Value.parse(request.testbed_signed, markers)
    logging.debug("Client's self-signed certificate name: %s", Name.to_str(client_self_signed.name))
    logging.debug("Client's testbed-signed certificate name: %s", Name.to_str(testbed_signed.name))
    
    # verify the interest signaure
    # we don't know the client public key until we parse the interest param
    asyncio.create_task(_validator_wrapper(name, sig_ptrs, verify_known_ecdsa_signature,
                                           on_interest_verify_success, on_interest_verify_failure, (markers, name)))

# this time we know the client public key   
@app.route('/edge/_ca/challenge', validator=verify_known_ecdsa_signature)
def on_challenge_interest(name: FormalName, param: InterestParam, _app_param: Optional[BinaryStr]):
    global client_self_signed, testbed_signed, signer, token
    logging.info("Challenge interest")

    # extract client public key bits
    client_pub_key = bytes(testbed_signed.content)

    # verify
    verifier = DSS.new(ECC.import_key(client_pub_key), 'fips-186-3', 'der')
    sha256_hash = SHA256.new()
    sha256_hash.update(token)
    try:
        verifier.verify(sha256_hash, bytes(_app_param))
    except ValueError:
        # proof-of-possesion failed
        logging.info("Proof-of-possesion failed")
        content = "proof-of-possesion failed".encode()
        app.put_data(name, content=content, freshness_period=10000, signer=signer)
        return

    # we can derive a new cert for client
    logging.debug("Proof-of-possesion succeed")
    client_key_name = client_self_signed.name[:-2]

    issuer_id = "icear-server"
    start_time = date.fromtimestamp(time.time())
    expire_sec = timedelta(days=30).total_seconds()

    cert_name, wire = derive_cert(client_key_name, issuer_id, client_pub_key, signer, start_time, expire_sec)
    logging.info("Newly issused certificate: %s", Name.to_str(cert_name))    
    app.put_data(name, content=wire, freshness_period=10000, signer=signer)

def bootstrap():
    global trust_anchor, signer
    import_safebag("sec/server.safebag", "1234")
    import_cert("sec/server.ndncert")
    
    with open("sec/server.safebag", "r") as safebag:
        wire = safebag.read()
        wire = base64.b64decode(wire)
        wire = parse_and_check_tl(wire, SecurityV2TypeNumber.SAFE_BAG)
        bag = SafeBag.parse(wire)
        testbed_signed = CertificateV2Value.parse(bag.certificate_v2)
        server_key_name = Name.to_str(testbed_signed.name[:-2])
        privateKey = serialization.load_der_private_key(bytes(bag.encrypted_key_bag), password=b'1234', backend=default_backend())
        server_prv_key = privateKey.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
        signer = Sha256WithEcdsaSigner(server_key_name, server_prv_key)

    with open("sec/testbed.anchor", "r") as ndncert:
        wire = ndncert.read()
        wire = base64.b64decode(wire)
        trust_anchor = parse_certificate(wire)

if __name__ == '__main__':
    # clear the environment just in case
    clear_server_keychain()
    bootstrap()
    app.run_forever()
    clear_server_keychain()

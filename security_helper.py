from ndn.encoding import TlvModel, BytesField, UintField, TypeNumber
import subprocess

class CertRequest(TlvModel):
    self_signed = BytesField(TypeNumber.DATA)
    testbed_signed = BytesField(TypeNumber.DATA)

# few command lines to bypass python-ndn keychain
def clear_client_keychain():
    try:
        subprocess.run(['ndnsec-delete', "/ndn/edu/ucla/cs/tianyuan-icear-client"], stdout=subprocess.PIPE)
        subprocess.run(['ndnsec-delete', "/ndn/edu/ucla/icear/tianyuan-icear-client"], stdout=subprocess.PIPE)
    except KeyError:
        pass

def clear_server_keychain():
    try:
        subprocess.run(['ndnsec-delete', "/ndn/edu/ucla/icear"], stdout=subprocess.PIPE)
    except KeyError:
        pass

def import_safebag(safebag, password):
    try:
        subprocess.run(['ndnsec-import', safebag, '-P', password], stdout=subprocess.PIPE)
    except KeyError:
        pass

def import_cert(cert):
    try:
        subprocess.run(['ndnsec-cert-install', cert], stdout=subprocess.PIPE)
    except KeyError:
        pass
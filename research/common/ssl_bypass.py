"""
ssl_bypass.py

Corporate TLS interception breaks certificate validation on this network, so
any HuggingFace access fails with CERTIFICATE_VERIFY_FAILED.

This bites us in one non-obvious place: RAM's `init_tokenizer()` loads
`bert-base-uncased` from the Hub every time the model is constructed. Even when
the tokenizer is already in the local cache, huggingface_hub revalidates over
the network — so a warm cache is NOT enough, and RAM++ load fails at runtime
with an SSLError that surfaces as an opaque Gradio "upstream app raised an
exception".

Same approach already used by research/vitb32_benchmark/app.py and
build_tinyclip_assets.py. Import this BEFORE torch/transformers/gradio:

    import ssl_bypass  # noqa: F401  (must precede transformers import)

Set HF_HUB_OFFLINE=1 in the environment to skip the network entirely and rely
on the cache, if you would rather not disable verification at all.
"""

from __future__ import annotations

import os
import ssl
import warnings

os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("SSL_CERT_FILE", "")
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERIFY_SSL", "0")

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:  # pragma: no cover
    pass

warnings.filterwarnings("ignore", message=".*verify.*")
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import requests as _rq

    _OrigSession = _rq.Session

    class _NoSSLSession(_OrigSession):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.verify = False

    _rq.Session = _NoSSLSession
    _rq.sessions.Session = _NoSSLSession
except ImportError:  # pragma: no cover
    pass

try:
    import httpx as _httpx

    _OC = _httpx.Client

    class _NoVerifyClient(_OC):
        def __init__(self, *a, **k):
            k["verify"] = False
            super().__init__(*a, **k)

    _httpx.Client = _NoVerifyClient

    _OA = _httpx.AsyncClient

    class _NoVerifyAsyncClient(_OA):
        def __init__(self, *a, **k):
            k["verify"] = False
            super().__init__(*a, **k)

    _httpx.AsyncClient = _NoVerifyAsyncClient
except ImportError:  # pragma: no cover
    pass

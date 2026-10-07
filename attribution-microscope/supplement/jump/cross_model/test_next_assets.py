import hashlib
from pathlib import Path
import tempfile
import unittest

from next_assets import identity


class IdentityTests(unittest.TestCase):
    def test_lfs_hash_and_exact_size_are_both_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model';p.write_bytes(b'fixed')
            spec=dict(size=5,lfs=dict(sha256=hashlib.sha256(b'fixed').hexdigest()))
            self.assertTrue(identity(p,spec))
            p.write_bytes(b'other');self.assertFalse(identity(p,spec))
            p.write_bytes(b'fixed-extra');self.assertFalse(identity(p,spec))

    def test_small_files_use_official_git_blob_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';p.write_bytes(b'{}')
            spec=dict(size=2,blobId=hashlib.sha1(b'blob 2\0{}').hexdigest())
            self.assertTrue(identity(p,spec))
            p.write_bytes(b'[]');self.assertFalse(identity(p,spec))

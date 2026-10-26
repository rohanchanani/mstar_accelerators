"""Resolve pinned platform assets through the standard Hub cache, then verify.

Locations are environment concerns (HF_HOME/HF_HUB_CACHE), not test config.
A cache miss is explicit missing evidence; no alternate revision is accepted.
"""
from pathlib import Path
import hashlib

from .harness import Unavailable


class AssetIntegrityError(ValueError):
    pass


def verify(path, entry):
    path = Path(path)
    size = path.stat().st_size
    if size != entry['bytes']:
        raise AssetIntegrityError(f'{path}: size differs from the pinned manifest')
    if 'sha256' in entry:
        digest, expected = hashlib.sha256(), entry['sha256']
    else:
        digest, expected = hashlib.sha1(), entry['git_blob_sha1']
        digest.update(f'blob {size}\0'.encode())
    with path.open('rb') as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise AssetIntegrityError(f'{path}: checksum differs from the pinned manifest')


def resolve(case):
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:
        raise Unavailable('asset_dependency', f'Hub cache resolver unavailable: {exc}') from exc
    manifest = case.manifest
    try:
        checkpoint = Path(snapshot_download(manifest['model'], revision=manifest['revision'],
                                           allow_patterns=[entry['path'] for entry in manifest['files']],
                                           local_files_only=True))
    except LocalEntryNotFoundError as exc:
        raise Unavailable('reference_assets_unavailable',
                          f"missing pinned Hub cache {manifest['model']}@{manifest['revision']}: {exc}") from exc
    identities = []
    for entry in manifest['files']:
        path = checkpoint / entry['path']
        if not path.is_file():
            raise Unavailable('reference_assets_unavailable', f'missing pinned asset {path}')
        verify(path, entry)
        identities.append(dict(entry))
    fixtures = {}
    if 'audio' in manifest:
        entry = manifest['audio']['derived']
        path = case.directory / entry['path']
        verify(path, entry)
        fixtures['audio'] = str(path)
        identities.append(dict(entry))
    return str(checkpoint), fixtures, {'checkpoint_path': str(checkpoint), 'verified': identities}

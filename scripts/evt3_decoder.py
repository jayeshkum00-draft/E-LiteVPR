"""Vectorized decoder for the Prophesee EVT3 event format.

Written to replace expelliarmus for real Prophesee streams: its EVT3
decoder adds TIME_LOW rollovers on top of the TIME_HIGH increments that
already encode them, doubling elapsed time (verified on NYC-Event-VPR
raws against a byte-level TIME_HIGH scan; matches open upstream issue
open-neuromorphic/expelliarmus#19). Its encoder writes t>>32 as
TIME_HIGH, so save/read round-trips are self-consistent and hide the
bug.

Format (16-bit little-endian words, type in bits [15:12]):
  0x0 EVT_ADDR_Y   : y address in [10:0]
  0x2 EVT_ADDR_X   : single CD event; x in [10:0], polarity in [11]
  0x3 VECT_BASE_X  : vector base; x in [10:0], polarity in [11]
  0x4 VECT_12      : events at base_x+bit for set bits [11:0]; base += 12
  0x5 VECT_8       : events at base_x+bit for set bits [7:0];  base += 8
  0x6 EVT_TIME_LOW : time[11:0]
  0x8 EVT_TIME_HIGH: time[23:12]; register decreases mark 24-bit rollover
  other types (EXT_TRIGGER 0xA, OTHERS 0xE, CONTINUED 0xF, ...) skipped.

Timestamp register model (as in Metavision): the decoder holds
(th_ovf, th, tl) and every event reads t = (th_ovf<<24)+(th<<12)+tl.
Events between a TIME_HIGH increment and the next TIME_LOW see a stale
TIME_LOW, which can step time backwards by <4096 us; output is clamped
monotonic and the worst correction is tracked (`max_backstep_us`). A
backstep >= 4096 us cannot be such a transient and raises.
"""
import numpy as np

_TY_Y, _TY_X, _TY_VB, _TY_V12, _TY_V8, _TY_TL, _TY_TH = (
    0x0, 0x2, 0x3, 0x4, 0x5, 0x6, 0x8)
_KNOWN_TYPES = (_TY_Y, _TY_X, _TY_VB, _TY_V12, _TY_V8, _TY_TL, _TY_TH)

def read_header_offset(f):
    """Skip the optional '%'-prefixed ASCII header; leave f at the data."""
    pos = 0
    f.seek(0)
    while True:
        line = f.readline()
        if line.startswith(b"%"):
            pos = f.tell()
        else:
            break
    f.seek(pos)
    return pos

class Evt3Decoder:
    """Stateful chunk decoder; feed consecutive word buffers of one file."""

    def __init__(self):
        self._th = 0                # TIME_HIGH register (12 bit)
        self._tl = 0                # TIME_LOW register (12 bit)
        self._ovf = 0               # TIME_HIGH rollover count
        self._y = 0                 # y register
        self._vx = 0                # next vector base x
        self._vp = 0                # vector polarity register
        self._last_t = 0            # monotonic clamp carry
        self.max_backstep_us = 0
        self.skipped_words = 0
        self.n_events = 0

    def decode(self, words):
        """words: uint16 array -> (x i16, y i16, t_us i64, p u8) arrays,
        t monotonically non-decreasing across all chunks of the file."""
        typ = (words >> np.uint16(12)).astype(np.uint8)
        val = words & np.uint16(0x0FFF)

        # --- TIME_HIGH register history (rollover-extended, us at bit 12)
        i_th = np.flatnonzero(typ == _TY_TH)
        th_vals = val[i_th].astype(np.int64)
        th_lut = np.empty(len(i_th) + 1, np.int64)
        th_lut[0] = (self._ovf << 24) + (self._th << 12)
        if th_vals.size:
            seq = np.concatenate(([self._th], th_vals))
            ovf = self._ovf + np.cumsum(np.diff(seq) < 0)
            th_lut[1:] = (ovf << 24) + (th_vals << 12)
            self._th, self._ovf = int(th_vals[-1]), int(ovf[-1])

        # --- TIME_LOW register history
        i_tl = np.flatnonzero(typ == _TY_TL)
        tl_vals = val[i_tl].astype(np.int64)
        tl_lut = np.concatenate(([np.int64(self._tl)], tl_vals))
        if tl_vals.size:
            self._tl = int(tl_vals[-1])

        # --- y register history
        i_y = np.flatnonzero(typ == _TY_Y)
        y_vals = (val[i_y] & np.uint16(0x7FF)).astype(np.int16)
        y_lut = np.concatenate(([np.int16(self._y)], y_vals))
        if y_vals.size:
            self._y = int(y_vals[-1])

        # --- single CD events
        i_x = np.flatnonzero(typ == _TY_X)
        sx = (val[i_x] & np.uint16(0x7FF)).astype(np.int16)
        sp = (val[i_x] >> np.uint16(11)).astype(np.uint8)

        # --- vector CD events: base x advances 12/8 per vector word and
        #     resets at each VECT_BASE_X; polarity comes from the base word
        i_vf = np.flatnonzero((typ == _TY_VB) | (typ == _TY_V12)
                              | (typ == _TY_V8))
        vf_typ, vf_val = typ[i_vf], val[i_vf]
        is_b = vf_typ == _TY_VB
        adv = np.zeros(len(i_vf), np.int64)
        adv[vf_typ == _TY_V12] = 12
        adv[vf_typ == _TY_V8] = 8
        excl = np.concatenate(([0], np.cumsum(adv)[:-1]))
        grp = np.cumsum(is_b)               # 0 = group carried from before
        b_pos = np.flatnonzero(is_b)
        starts = np.concatenate(
            ([np.int64(self._vx)], (vf_val[b_pos] & 0x7FF).astype(np.int64)))
        pols = np.concatenate(
            ([np.int64(self._vp)], (vf_val[b_pos] >> 11).astype(np.int64)))
        excl0 = np.concatenate(([0], excl[b_pos]))
        vf_x0 = starts[grp] + (excl - excl0[grp])
        vf_p = pols[grp]
        if len(i_vf):
            self._vx = int(vf_x0[-1] + adv[-1])
            self._vp = int(vf_p[-1])

        def expand(want_typ, nbits):
            m = np.flatnonzero(vf_typ == want_typ)
            if not m.size:
                z = np.empty(0, np.int64)
                return z, z.astype(np.int16), z.astype(np.uint8)
            bits = (vf_val[m, None] >> np.arange(nbits, dtype=np.uint16)) & 1
            w_rel, bit = np.nonzero(bits)
            x = (vf_x0[m][w_rel] + bit).astype(np.int16)
            p = vf_p[m][w_rel].astype(np.uint8)
            return i_vf[m][w_rel], x, p

        w12, x12, p12 = expand(_TY_V12, 12)
        w8, x8, p8 = expand(_TY_V8, 8)

        # --- merge event sources in stream (word) order
        w = np.concatenate([i_x, w12, w8])
        x = np.concatenate([sx, x12, x8])
        p = np.concatenate([sp, p12, p8])
        order = np.argsort(w, kind="stable")
        w, x, p = w[order], x[order], p[order]

        y = y_lut[np.searchsorted(i_y, w, side="right")]
        t = (th_lut[np.searchsorted(i_th, w, side="right")]
             + tl_lut[np.searchsorted(i_tl, w, side="right")])

        if t.size:
            tm = np.maximum.accumulate(
                np.concatenate(([np.int64(self._last_t)], t)))[1:]
            back = int((tm - t).max())
            if back >= 4096:
                raise RuntimeError(
                    f"time reversal of {back} us -- larger than a stale-"
                    "TIME_LOW transient; stream is not well-formed EVT3")
            if back > self.max_backstep_us:
                self.max_backstep_us = back
            self._last_t = int(tm[-1])
            t = tm

        self.skipped_words += int(
            np.isin(typ, _KNOWN_TYPES, invert=True).sum())
        self.n_events += len(w)
        return x, y, t, p

def open_evt3(path):
    """Open a .raw, or stream the single .raw member of a .zip without
    extracting it (NYC-Event-VPR ships one ~44 GB raw deflated per zip)."""
    if str(path).lower().endswith(".zip"):
        import zipfile
        zf = zipfile.ZipFile(path)
        raws = [n for n in zf.namelist() if n.lower().endswith(".raw")]
        if len(raws) != 1:
            raise RuntimeError(
                f"{path}: expected exactly one .raw member, got {raws}")
        return zf.open(raws[0])
    return open(path, "rb")


def stream_evt3(path, words_per_chunk=1 << 24, decoder=None):
    """Yield (x, y, t_us, p) chunks for one .raw (or zipped .raw) file,
    t_us int64 relative to the recording. Pass an Evt3Decoder to inspect
    stats afterwards."""
    dec = decoder if decoder is not None else Evt3Decoder()
    with open_evt3(path) as f:
        read_header_offset(f)
        while True:
            buf = f.read(words_per_chunk * 2)
            if not buf:
                break
            if len(buf) % 2:            # truncated trailing byte
                buf = buf[:-1]
                if not buf:
                    break
            x, y, t, p = dec.decode(np.frombuffer(buf, dtype="<u2"))
            if x.size:
                yield x, y, t, p

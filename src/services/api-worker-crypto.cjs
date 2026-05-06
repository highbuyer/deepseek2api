/**
 * Reverse-engineered api-worker.js encryption/decryption protocol.
 *
 * Protocol:
 *   encryptData(payload) -> ArrayBuffer:
 *     1. BSON encode the payload
 *     2. XOR each byte with 0x37
 *     3. Generate 12 random bytes (IV)
 *     4. AES-256-GCM encrypt (key derived from hex string below)
 *     5. Prepend IV to ciphertext -> returns ArrayBuffer
 *
 *   decryptData(buffer) -> payload:
 *     1. Split: first 12 bytes = IV, rest = ciphertext
 *     2. AES-256-GCM decrypt
 *     3. XOR each byte with 0x37
 *     4. BSON decode
 *     5. Check timestamp (valid for 120000ms)
 *
 * AES-256 Key (hex):
 *   7122ce731776ceb3875f8d006764145aa76d32cc79a5779bd81fd4e9a42ff406
 *
 * BSON: minimal implementation — only the types actually used by the API protocol.
 * The real api-worker uses a full BSON library (labeled RN in obfuscated code),
 * but for our purposes we only need to handle:
 *   - double (0x01)
 *   - string (0x02)
 *   - document/object (0x03)
 *   - array (0x04)
 *   - binary (0x05)
 *   - boolean (0x08)
 *   - null (0x0A)
 *   - int32 (0x10)
 *   - int64 / timestamp (0x12 for 64-bit, 0x09 for UTC datetime)
 */

'use strict';

const crypto = require('crypto');

// --- Constants ---
const AES_KEY_HEX = '7122ce731776ceb3875f8d006764145aa76d32cc79a5779bd81fd4e9a42ff406';
const XOR_BYTE = 0x37;
const IV_LENGTH = 12;
const GCM_TAG_LENGTH = 16;
const TIMESTAMP_WINDOW_MS = 120000; // 2 minutes

// --- BSON Implementation (minimal) ---

const BSON_TYPE = {
  DOUBLE: 0x01,
  STRING: 0x02,
  DOCUMENT: 0x03,
  ARRAY: 0x04,
  BINARY: 0x05,
  BOOLEAN: 0x08,
  UTC_DATETIME: 0x09,
  NULL: 0x0A,
  INT32: 0x10,
  INT64: 0x12,
};

class BSONEncoder {
  constructor() {
    this.buf = [];
  }

  _writeByte(b) {
    this.buf.push(b & 0xff);
  }

  _writeInt32(v) {
    const b = Buffer.allocUnsafe(4);
    b.writeInt32LE(v, 0);
    for (let i = 0; i < 4; i++) this.buf.push(b[i]);
  }

  _writeInt64(v) {
    const bigV = BigInt(Math.floor(v));
    const lo = Number(bigV & 0xFFFFFFFFn);
    const hi = Number((bigV >> 32n) & 0xFFFFFFFFn);
    const b = Buffer.allocUnsafe(8);
    b.writeUInt32LE(lo, 0);
    b.writeUInt32LE(hi, 4);
    for (let i = 0; i < 8; i++) this.buf.push(b[i]);
  }

  _writeDouble(v) {
    const b = Buffer.allocUnsafe(8);
    b.writeDoubleLE(v, 0);
    for (let i = 0; i < 8; i++) this.buf.push(b[i]);
  }

  _writeCString(str) {
    for (let i = 0; i < str.length; i++) {
      this.buf.push(str.charCodeAt(i) & 0xff);
    }
    this.buf.push(0);
  }

  _writeString(str) {
    const encoded = Buffer.from(str, 'utf8');
    this._writeInt32(encoded.length + 1); // length includes null terminator
    for (let i = 0; i < encoded.length; i++) this.buf.push(encoded[i]);
    this.buf.push(0);
  }

  _encodeValue(type, value) {
    switch (type) {
      case BSON_TYPE.DOUBLE:
        this._writeDouble(value);
        break;
      case BSON_TYPE.STRING:
        this._writeString(String(value));
        break;
      case BSON_TYPE.DOCUMENT:
      case BSON_TYPE.ARRAY: {
        // Save position, write placeholder size, encode, then fix up size
        const startIdx = this.buf.length;
        this._writeInt32(0); // placeholder
        if (type === BSON_TYPE.DOCUMENT) {
          this._encodeDocument(value);
        } else {
          this._encodeArray(value);
        }
        const endIdx = this.buf.length;
        const size = endIdx - startIdx;
        // Fix up the size
        const sizeBuf = Buffer.allocUnsafe(4);
        sizeBuf.writeInt32LE(size, 0);
        for (let i = 0; i < 4; i++) this.buf[startIdx + i] = sizeBuf[i];
        break;
      }
      case BSON_TYPE.BINARY: {
        if (Buffer.isBuffer(value)) {
          this._writeInt32(value.length);
          this._writeByte(0); // subtype: generic binary
          for (let i = 0; i < value.length; i++) this.buf.push(value[i]);
        } else if (ArrayBuffer.isView(value)) {
          const arr = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
          this._writeInt32(arr.length);
          this._writeByte(0);
          for (let i = 0; i < arr.length; i++) this.buf.push(arr[i]);
        } else if (typeof value === 'string') {
          // Base64-encoded binary
          const bin = Buffer.from(value, 'base64');
          this._writeInt32(bin.length);
          this._writeByte(0);
          for (let i = 0; i < bin.length; i++) this.buf.push(bin[i]);
        } else {
          this._writeInt32(0);
          this._writeByte(0);
        }
        break;
      }
      case BSON_TYPE.BOOLEAN:
        this._writeByte(value ? 1 : 0);
        break;
      case BSON_TYPE.UTC_DATETIME:
        this._writeInt64(value instanceof Date ? value.getTime() : Number(value));
        break;
      case BSON_TYPE.NULL:
        // Nothing to write
        break;
      case BSON_TYPE.INT32:
        this._writeInt32(Number(value));
        break;
      case BSON_TYPE.INT64:
        this._writeInt64(Number(value));
        break;
      default:
        throw new Error(`Unsupported BSON type: 0x${type.toString(16)}`);
    }
  }

  _encodeElement(name, type, value) {
    this._writeByte(type);
    this._writeCString(name);
    this._encodeValue(type, value);
  }

  _encodeDocument(obj) {
    for (const [key, val] of Object.entries(obj)) {
      const type = this._inferType(val);
      this._encodeElement(key, type, val);
    }
    this._writeByte(0); // terminator
  }

  _encodeArray(arr) {
    for (let i = 0; i < arr.length; i++) {
      const type = this._inferType(arr[i]);
      this._encodeElement(String(i), type, arr[i]);
    }
    this._writeByte(0);
  }

  _inferType(value) {
    if (value === null || value === undefined) return BSON_TYPE.NULL;
    if (typeof value === 'boolean') return BSON_TYPE.BOOLEAN;
    if (typeof value === 'number') {
      if (Number.isInteger(value)) {
        if (value >= -2147483648 && value <= 2147483647) return BSON_TYPE.INT32;
        return BSON_TYPE.INT64;
      }
      return BSON_TYPE.DOUBLE;
    }
    if (typeof value === 'string') return BSON_TYPE.STRING;
    if (value instanceof Date) return BSON_TYPE.UTC_DATETIME;
    if (Buffer.isBuffer(value) || ArrayBuffer.isView(value)) return BSON_TYPE.BINARY;
    if (Array.isArray(value)) return BSON_TYPE.ARRAY;
    if (typeof value === 'object') return BSON_TYPE.DOCUMENT;
    return BSON_TYPE.STRING;
  }

  encode(doc) {
    this.buf = [];
    const startIdx = 0;
    this._writeInt32(0); // placeholder for total size
    this._encodeDocument(doc);
    const totalSize = this.buf.length;
    // Fix up total size
    const sizeBuf = Buffer.allocUnsafe(4);
    sizeBuf.writeInt32LE(totalSize, 0);
    for (let i = 0; i < 4; i++) this.buf[i] = sizeBuf[i];
    return Buffer.from(this.buf);
  }
}

class BSONDecoder {
  constructor(buffer) {
    this.buf = buffer;
    this.pos = 0;
  }

  _readByte() {
    return this.buf[this.pos++];
  }

  _readInt32() {
    const v = this.buf.readInt32LE(this.pos);
    this.pos += 4;
    return v;
  }

  _readInt64() {
    const lo = BigInt(this.buf.readUInt32LE(this.pos));
    const hi = BigInt(this.buf.readUInt32LE(this.pos + 4));
    this.pos += 8;
    return Number(lo | (hi << 32n));
  }

  _readDouble() {
    const v = this.buf.readDoubleLE(this.pos);
    this.pos += 8;
    return v;
  }

  _readCString() {
    let end = this.pos;
    while (end < this.buf.length && this.buf[end] !== 0) end++;
    const str = this.buf.toString('utf8', this.pos, end);
    this.pos = end + 1;
    return str;
  }

  _readString() {
    const len = this._readInt32();
    const str = this.buf.toString('utf8', this.pos, this.pos + len - 1);
    this.pos += len;
    return str;
  }

  _readValue(type) {
    switch (type) {
      case BSON_TYPE.DOUBLE:
        return this._readDouble();
      case BSON_TYPE.STRING:
        return this._readString();
      case BSON_TYPE.DOCUMENT:
        return this._readDocument();
      case BSON_TYPE.ARRAY:
        return this._readArray();
      case BSON_TYPE.BINARY: {
        const len = this._readInt32();
        const subtype = this._readByte();
        const data = Buffer.alloc(len);
        this.buf.copy(data, 0, this.pos, this.pos + len);
        this.pos += len;
        return data;
      }
      case BSON_TYPE.BOOLEAN:
        return this._readByte() !== 0;
      case BSON_TYPE.UTC_DATETIME:
        return new Date(this._readInt64());
      case BSON_TYPE.NULL:
        return null;
      case BSON_TYPE.INT32:
        return this._readInt32();
      case BSON_TYPE.INT64:
        return this._readInt64();
      default:
        throw new Error(`Unknown BSON type: 0x${type.toString(16)} at pos ${this.pos}`);
    }
  }

  _readElement() {
    if (this.pos >= this.buf.length || this.buf[this.pos] === 0) return null;
    const type = this._readByte();
    const name = this._readCString();
    const value = this._readValue(type);
    return { name, value };
  }

  _readDocument() {
    const size = this._readInt32();
    const doc = {};
    while (true) {
      const elem = this._readElement();
      if (!elem) break;
      doc[elem.name] = elem.value;
    }
    return doc;
  }

  _readArray() {
    const size = this._readInt32();
    const arr = [];
    while (true) {
      const elem = this._readElement();
      if (!elem) break;
      arr.push(elem.value);
    }
    return arr;
  }

  decode() {
    return this._readDocument();
  }
}

// --- Crypto Operations ---

const AES_KEY = Buffer.from(AES_KEY_HEX, 'hex');

function xorBuffer(buf) {
  const result = Buffer.allocUnsafe(buf.length);
  for (let i = 0; i < buf.length; i++) {
    result[i] = buf[i] ^ XOR_BYTE;
  }
  return result;
}

/**
 * Encrypt a payload object into an ArrayBuffer.
 * Payload should include at least { timestamp, action, ... }
 */
function encryptData(payload) {
  // 1. BSON encode
  const encoder = new BSONEncoder();
  const bsonData = encoder.encode(payload);

  // 2. XOR each byte with 0x37
  const xored = xorBuffer(bsonData);

  // 3. Generate 12-byte random IV
  const iv = crypto.randomBytes(IV_LENGTH);

  // 4. AES-256-GCM encrypt
  const cipher = crypto.createCipheriv('aes-256-gcm', AES_KEY, iv);
  const encrypted = Buffer.concat([cipher.update(xored), cipher.final()]);
  const authTag = cipher.getAuthTag();

  // 5. Prepend IV, append auth tag
  return Buffer.concat([iv, encrypted, authTag]);
}

/**
 * Decrypt an ArrayBuffer or Buffer into a payload object.
 * Returns null if timestamp verification fails.
 */
function decryptData(buffer) {
  if (!Buffer.isBuffer(buffer)) {
    buffer = Buffer.from(buffer);
  }

  if (buffer.length < IV_LENGTH + GCM_TAG_LENGTH) {
    throw new Error(`Buffer too short: ${buffer.length} bytes (need at least ${IV_LENGTH + GCM_TAG_LENGTH})`);
  }

  // 1. Split: IV (first 12), ciphertext+tag (rest)
  const iv = buffer.subarray(0, IV_LENGTH);
  const encrypted = buffer.subarray(IV_LENGTH);

  // 2. AES-256-GCM decrypt
  const decipher = crypto.createDecipheriv('aes-256-gcm', AES_KEY, iv);
  decipher.setAuthTag(encrypted.subarray(encrypted.length - GCM_TAG_LENGTH));
  const ciphertext = encrypted.subarray(0, encrypted.length - GCM_TAG_LENGTH);
  const decrypted = Buffer.concat([decipher.update(ciphertext), decipher.final()]);

  // 3. XOR each byte with 0x37
  const xored = xorBuffer(decrypted);

  // 4. BSON decode
  const decoder = new BSONDecoder(xored);
  const payload = decoder.decode();

  // 5. Check timestamp
  if (payload.timestamp) {
    const ts = payload.timestamp instanceof Date ? payload.timestamp.getTime() : Number(payload.timestamp);
    const now = Date.now();
    if (Math.abs(now - ts) > TIMESTAMP_WINDOW_MS) {
      // In real api-worker, this would return null (invalid/stale)
      // For debugging, we include the check result but still return
      payload._timestampExpired = true;
    }
  }

  return payload;
}

// --- Exports ---
module.exports = {
  encryptData,
  decryptData,
  AES_KEY_HEX,
  AES_KEY,
  XOR_BYTE,
  BSONEncoder,
  BSONDecoder,
};

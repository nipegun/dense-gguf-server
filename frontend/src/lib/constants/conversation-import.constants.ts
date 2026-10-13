// First bytes of every ZIP local file header ("PK"). Import detects an archive
// from these bytes rather than from the filename, which the OS may not preserve.
export const aZipMagic = [0x50, 0x4b];

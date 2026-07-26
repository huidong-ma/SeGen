<div align="center">
<h1>SeGen: Neural Lossless Genomic Compression with Sensitive-Region Protection</h1>
</div>

## 📄 Introduction
SeGen is a learning-based framework for lossless genomic compression with selective sensitive-region protection. It is designed for deployments where a data holder may outsource compression to an untrusted service without exposing dictionary-defined sensitive genomic content. Rather than encrypting the entire sequence before compression, SeGen uses a selective masking and compression workflow: the data holder identifies sensitive regions, masks only those bases, protects the location metadata and recovery key, and then compresses the masked sequence.

The masking operation preserves sequence length and the four-symbol DNA alphabet `{A, C, G, T}`, so the protected sequence remains compatible with DNA grouping, GCNN probability prediction, and arithmetic coding. An authorized recipient verifies the archive, recovers the sensitive-region bitmap, and inversely masks the sequence to reconstruct the original DNA exactly.

<div align="center">
  <picture>
    <img src="https://github.com/huidong-ma/SeGen/blob/main/assets/framework.png" width="800" alt="Framework of proposed method.">
  </picture>
  <br>
  <b>Framework of proposed method.</b>
</div>

Key properties of the current implementation include:
* Selective confidentiality for dictionary-defined sensitive regions, with AES-CTR masking over the four-base alphabet.
* Encrypted interval metadata using AES-256-GCM, RSA-OAEP-SHA-256 key wrapping, and HMAC-SHA-256 archive authentication.
* DNA Grouper and an online GCNN with arithmetic coding, allowing compression without access to the recovery key or sensitive-location metadata.
* End-to-end lossless recovery after authenticated decompression.


<div align="center">
  <picture>
    <img src="https://github.com/huidong-ma/SeGen/blob/main/assets/results.png" width="800" alt="Performance of all methods.">
  </picture>
  <br>
  <b>Overall performance of all method on real-world datasets.</b>
</div>
## 💡 Usage

### I. Setup

1. Enter the project directory and use the included executables:

    ```bash
    cd SeGen2
    chmod +x ./segen PRSEC/prsec PRSEC/skmer
    ```

   The repository already includes compiled `PRSEC/prsec` and `PRSEC/skmer` binaries, so no compilation is required for a normal run. Rebuild only after changing the PSSM source or when the binaries are unavailable or incompatible:

    ```bash
    make -C PRSEC
    ```

   Rebuilding requires a C++17 compiler, OpenMP, and OpenSSL development libraries.

2. Create and activate a Conda environment:

   ```bash
   conda create -n segen_env python=3.12 -y
   conda activate segen_env
   ```

3. Install NumPy and PyTorch.

   Install a PyTorch version compatible with the CUDA driver and GPU used for compression and decompression. Consult the official PyTorch installation selector:
   https://pytorch.org/get-started/previous-versions/

   For example, choose the proper command for your CUDA version from the PyTorch website and run it inside the `segen_env` environment.

4. Download the sensitive sequence database ([`dataBaseSrf.tar.gz`](https://drive.google.com/file/d/1ER9jGTI2UmBj_coyZ2Xm9TUsv7gb8Rkk/view?usp=drive_link)) and extract it:

   ```bash
   tar -zxvf dataBaseSrf.tar.gz
   ```

   After extraction, make sure `dataBaseSrf.txt` exists in the SeGen project directory.

5. Generate the recipient RSA-3072 key pair on the first setup only. Skip this step when `recipient_private.pem` and `recipient_public.pem` already exist:

   ```bash
   openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out recipient_private.pem
   openssl pkey -in recipient_private.pem -pubout -out recipient_public.pem
   chmod 600 recipient_private.pem
   ```

   Neither key needs to be uploaded to the repository. Keep `recipient_private.pem` local and never commit or share it; only `recipient_public.pem` needs to be sent to the data holder when compression and recovery are performed by different parties. The public key is used for compression, and the private key is required for authenticated recovery. The RSA key pair can be reused; SeGen automatically generates a fresh random session key for every archive.

### II. Compression and Decompression
The raw input must be a single non-empty line containing only uppercase `A`, `C`, `G`, and `T`. FASTA headers, multiline FASTA files, whitespace, and ambiguous bases such as `N` are not accepted by the current PSSM executable.

Compress a raw genomic sequence file:

```bash
./segen c <raw_file> <compressed_file>
```

Decompress a compressed file:

```bash
./segen d <compressed_file> <decompressed_file>
```

**Example:**

```bash
./segen c test test.cmp
./segen d test.cmp test.decmp
```

To verify lossless reconstruction, compare the original file with the decompressed file:

```bash
cmp test test.decmp
```

If there is no output from `cmp`, the decompressed file is identical to the original file.

---

### III. Command reference

```bash
./segen c <raw_file> <archive_file> [recipient_public.pem] [threads] [K] [S] [GPU] [vocab_size]
./segen d <archive_file> <recovered_file> [recipient_private.pem] [threads] [K] [S] [GPU] [vocab_size]
```

Defaults are `threads=32`, `K=3`, `S=3`, `GPU=0`, and `vocab_size=64`. `S` must be no greater than `K`, and `vocab_size` must be at least `4^K`. Compression and decompression must use the same values. If the key path is omitted, the corresponding key beside `segen` is used.

The standard commands are:

```bash
./segen c test test.cmp recipient_public.pem 32 3 3 0 64
./segen d test.cmp test.decmp recipient_private.pem 32 3 3 0 64

cmp test test.decmp
```

A wrong private key or modified protected archive causes recovery to fail without writing the recovered plaintext. The archive contains the compressed sequence, compression parameters, encrypted sensitive-region metadata, RSA-wrapped session key, and HMAC tag.

## 📦 Dataset
The datasets used in the paper can be directly downloaded from [datasets.tar.gz](https://drive.google.com/file/d/1LShFvdYzGvXiFhzEU7dZUfEx-jPFSlDO/view?usp=drive_link) and extracted by executing `tar -xzf datasets.tar.gz`. The detailed information is as follows.
| **Name** | **Size (B)** | **Entropy** | **Description** |
| :---: | :---: | :---: | :---: |
| **AeCa** | 1,591,049 | 1.987 | A medium-sized aeropyrum camini dataset |
| **HePy** | 1,667,825 | 1.964 | A medium-sized helicobacter pylori bacterial dataset |
| **HaHi** | 3,890,005 | 1.955 | A large-scale haloarcula hispanica archaea dataset |
| **EsCo** | 4,641,652 | 2.000 | A medium-sized escherichia coli bacteria dataset |
| **SnSt** | 6,254,100 | 1.983 | A small-sized Human high-level privacy-sensitive dataset with STRs and SNPs |
| **PlFa** | 8,986,712 | 1.974 | A small-scale plasmodium falciparum dataset |
| **WaMe** | 9,144,432 | 1.976 | A GD dataset of unknown species |
| **ScPo** | 10,652,155 | 1.964 | A medium-sized schizosaccharomyces pombe dataset |
| **HuMa** | 16,577,023 | 1.931 | A dataset of human genome data |
| **EnIn** | 26,403,087 | 1.951 | A medium-sized entamoeba invadens dataset |
| **DrMe** | 32,181,429 | 1.991 | A medium-sized chromosome-2 of the drosophila miranda |
| **OrSa** | 43,262,523 | 1.989 | A large chromosome-1 dataset of oryza sativa japonica |
| **DaRe** | 62,565,020 | 1.950 | The chromosome-3 of the danio rerio |
| **AnCa** | 142,189,675 | 1.968 | One genome dataset of unknown species |
| **GaGa** | 148,532,294 | 1.970 | A large-scale chromosome-2 of the gallus gallus |
| **HoSa** | 189,752,667 | 1.960 | A large-scale human genome dataset |


---

## 🔥 Change Logs
- *2025.07.30*: Fixed several bugs. SeGen is now more user-friendly. 
- *2025.05.20*: Initial bug fixes and improvements.

---

## 📖 Acknowledgment
The code is based on [PAC](https://github.com/mynotwo/Faster-and-Stronger-Lossless-Compression-with-Optimized-Autoregressive-Framework), [MSDZip](https://github.com/mhuidong/MSDZip), and [Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding). Thanks for these great works!

## ✉️ Contact
Email: mahd@nbjl.nankai.edu.cn  
Nankai-Baidu Joint Laboratory (NBJL)

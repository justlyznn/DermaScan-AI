from huggingface_hub import HfApi
import getpass
import sys

print("===========================================")
print("🚀 DermaScan AI - Hugging Face Uploader 🚀")
print("===========================================\n")

print("Silakan ambil Token Hugging Face Anda yang bertipe 'Write'")
token = getpass.getpass("Paste Token Anda di sini (lalu tekan Enter): ")

api = HfApi(token=token)

print("\nMengunggah file ke Hugging Face... (Termasuk model AI 200MB)")
print("Mohon tunggu beberapa menit, jangan tutup jendela ini...\n")

try:
    api.upload_folder(
        folder_path=".",
        repo_id="adityaulil/DermaScan_AI",
        repo_type="space",
        ignore_patterns=[".git/*", ".github/*", "upload_to_hf.py", "data/*", "notebooks/*", "outputs/*", "logs_kaggle_running.txt"]
    )
    print("\n✅ SUKSES! Semua file berhasil diunggah dengan sempurna.")
    print("🌐 Cek aplikasi Anda di: https://huggingface.co/spaces/adityaulil/DermaScan_AI")
except Exception as e:
    print("\n❌ GAGAL MENGUNGGAH. Error:", str(e))
    print("Pastikan Token Anda memiliki akses 'Write'!")

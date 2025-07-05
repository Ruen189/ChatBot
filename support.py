from huggingface_hub import snapshot_download
# repo_id может отличаться — убедитесь на HF, что именно называется Llama-3.2-1B
local_folder = snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B",
    use_auth_token=True
)
print("Модель скачана в:", local_folder)
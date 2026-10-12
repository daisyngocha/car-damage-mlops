# Hướng dẫn dùng Claude Code cho repo car-damage-mlops

Ghi chú cá nhân để lần sau mở lại vẫn làm được ngay. Repo: `daisyngocha/car-damage-mlops`.

---

## 1. Bức tranh tổng thể: code nằm ở đâu, ai sửa được

Có **3 nơi** chứa code, mỗi nơi là một bản độc lập. Cầu nối duy nhất giữa chúng là **GitHub** (push / pull).

| Nơi | Đường dẫn | Ai làm việc trên đó | Thấy data/model local? |
|---|---|---|---|
| WSL (máy mình) – **bản chính** | `~/projects/car-damage-mlops` = `/home/daisyngocha/projects/car-damage-mlops` | Claude Code CLI chạy trong terminal WSL | Có |
| GitHub | github.com/daisyngocha/car-damage-mlops | Nơi trung gian, CI chạy pytest | Không (data bị `.gitignore`) |
| Cloud session (claude.ai/code, app Claude) | container trên cloud, ví dụ `/home/user/car-damage-mlops` | Claude chạy trên server Anthropic, clone từ GitHub | Không |

Hiểu nhầm cần tránh:

- Claude Code CLI **chạy được trong WSL** và sửa trực tiếp file trong WSL. Không phải sửa tay.
- WSL và Windows **không bị cô lập**: từ WSL thấy ổ C qua `/mnt/c/...`, từ Windows thấy WSL qua `\\wsl$\Ubuntu\home\daisyngocha\...`.
- Cloud session **không kết nối được** vào máy mình. Muốn nó thấy thay đổi local thì phải push lên GitHub. Muốn máy mình thấy thay đổi cloud thì phải `git pull`.
- Chỉ giữ **một bản repo trên máy**, đặt trong WSL. Không giữ thêm bản ở `C:\...` để tránh lệch code. Chạy Python/uv/DVC trên file `/mnt/c` cũng chậm hơn nhiều.

---

## 2. Trường hợp chính: Claude Code CLI trong WSL

### 2.1. Cài đặt (làm 1 lần)

Trong terminal WSL (Ubuntu):

```bash
# Cách 1: native installer (khuyến nghị)
curl -fsSL https://claude.ai/install.sh | bash

# Cách 2: qua npm (cần Node.js >= 18)
npm install -g @anthropic-ai/claude-code

claude --version     # kiểm tra
claude update        # cập nhật bản mới
```

### 2.2. Đăng nhập (làm 1 lần, hoặc khi hết phiên)

1. Gõ `claude`. Màn hình hiện URL đăng nhập dạng `https://claude.com/cai/oauth/authorize?...`.
2. Nếu trình duyệt không tự mở: nhấn `c` để copy URL, dán vào trình duyệt bên Windows.
3. Đăng nhập tài khoản Claude (gói Pro), bấm **Authorize**.
4. Trang callback hiện **authorization code**. Copy code, dán vào dòng `Paste code here if prompted >` rồi Enter.

Lưu ý:

- URL gắn với đúng phiên terminal đó. Thoát CLI hoặc code hết hạn thì chạy lại `claude` (hoặc `/login`) để lấy URL mới.
- **Không gửi code cho ai, không dán vào chat.**
- Đổi tài khoản: `/logout` rồi `/login`.

### 2.3. Mỗi lần bắt đầu làm việc

```bash
cd ~/projects/car-damage-mlops
git checkout main && git pull        # đồng bộ với GitHub trước
uv sync --locked                     # đảm bảo .venv đúng uv.lock
claude                               # mở Claude Code
```

Lần đầu mở một thư mục sẽ có câu hỏi **"Is this a project you created or one you trust?"**:

- Repo của mình → chọn **Yes, I trust this folder**.
- Repo lạ → chọn No, kiểm tra trước `.claude/settings.json` (hooks tự chạy lệnh) và `.mcp.json` (MCP server), vì trust xong Claude Code sẽ tự load các file này.

Tiếp tục cuộc trò chuyện cũ thay vì mở mới:

```bash
claude -c            # tiếp tục phiên gần nhất trong thư mục này
claude --resume      # chọn phiên cũ từ danh sách
```

### 2.4. Màn hình CLI và các chế độ quyền (permission mode)

Dòng dưới cùng hiển thị chế độ hiện tại. Nhấn **Shift+Tab** để chuyển vòng:

| Chế độ | Hành vi | Dùng khi |
|---|---|---|
| default | Hỏi trước khi sửa file và trước khi chạy lệnh | Mới học, muốn duyệt từng bước |
| accept edits | Tự sửa file, vẫn hỏi trước khi chạy lệnh shell | Thường dùng nhất khi đã quen |
| plan | Chỉ đọc và lập kế hoạch, không sửa gì | Việc lớn: refactor, thêm DVC/Docker. Duyệt plan rồi mới cho làm |
| auto | Claude tự đánh giá rủi ro: lệnh an toàn tự chạy, lệnh nguy hiểm bị chặn | Việc lặp lại, đã tin tưởng luồng làm |

Khi Claude hỏi quyền: chọn Yes (1 lần), Yes and don't ask again (cho lệnh đó về sau), hoặc No và gõ lại hướng dẫn.

### 2.5. Phím tắt và lệnh hay dùng

| Gõ | Tác dụng |
|---|---|
| `! <lệnh>` | Chạy lệnh shell trực tiếp, ví dụ `! git status`, `! uv run pytest -q` |
| `@đường/dẫn` | Đưa file vào ngữ cảnh, ví dụ `@src/train.py giải thích hàm này` |
| `Esc` | Dừng Claude giữa chừng (khi thấy đi sai hướng) |
| `Esc` `Esc` | Quay lại tin nhắn trước để sửa prompt hoặc rewind |
| `Shift+Tab` | Đổi permission mode |
| `/init` | Tạo `CLAUDE.md`: mô tả repo, cách chạy, quy ước. Commit file này |
| `/memory` | Mở và sửa `CLAUDE.md` / memory |
| `/clear` | Xoá ngữ cảnh, bắt đầu việc mới (tránh lẫn việc cũ, tiết kiệm quota) |
| `/compact` | Tóm tắt hội thoại dài để giảm ngữ cảnh |
| `/model` | Đổi model |
| `/usage` | Xem mức dùng so với hạn mức gói Pro |
| `/permissions` | Xem / sửa các lệnh đã cho phép |
| `/remote-control` | Điều khiển phiên này từ app Claude (xem mục 4) |
| `/help` | Danh sách lệnh |
| `/exit` hoặc `Ctrl+C` 2 lần | Thoát |

### 2.6. Quy trình chuẩn cho một thay đổi (feature branch → PR)

Không push thẳng lên `main`. CI (`.github/workflows/tests.yml`) chạy pytest trên mọi PR.

```
1. ! git checkout main && git pull
2. ! git checkout -b feat/<tên-việc>
3. Prompt cho Claude (để ở plan mode nếu việc lớn)
4. Duyệt diff: ! git diff
5. Kiểm tra: ! uv run pytest -q   và   ! uv run ruff check .
6. Prompt: "commit với message rõ ràng và push branch này lên origin"
7. Tạo PR trên GitHub (hoặc nhờ Claude nếu đã cài gh: "tạo PR vào main")
8. Đợi CI xanh → merge → về main: ! git checkout main && git pull
```

Push lần đầu cần xác thực GitHub trong WSL, chọn một trong hai:

```bash
# Cách A: GitHub CLI
sudo apt install gh && gh auth login
# Cách B: SSH key
ssh-keygen -t ed25519 -C "email@...";  cat ~/.ssh/id_ed25519.pub   # dán vào GitHub > Settings > SSH keys
git remote set-url origin git@github.com:daisyngocha/car-damage-mlops.git
```

### 2.7. Mẫu prompt dùng được ngay cho repo này

```
chạy pytest và báo test nào fail, giải thích nguyên nhân, chưa sửa gì
```
```
@src/train.py @conf/config.yaml giải thích luồng train từ Hydra config đến lúc lưu checkpoint
```
```
README đang lỗi thời (vẫn nhắc main.py, ImageNet). Viết lại cho đúng: setup bằng uv,
chạy src.train / src.evaluate với Hydra override, chạy test. Giữ ghi chú license BSD-3.
```
```
Hướng dẫn tôi thêm DVC cho data/raw từng bước: dvc init, track data/raw, chọn remote,
cập nhật .gitignore. Làm trên branch feat/dvc, giải thích từng lệnh trước khi chạy.
```
```
train thử 1 epoch với override hydra train.epochs=1, rồi so sánh metrics với reports/metrics.json
```

Mẹo viết prompt:

- Nêu rõ **phạm vi** (file nào, branch nào) và **điều kiện xong** (test pass, không đổi API...).
- Muốn học thì thêm "giải thích trước khi làm" hoặc dùng plan mode.
- Mỗi việc một phiên. Xong việc thì `/clear`.

---

## 3. Trường hợp: thư mục nằm bên Windows (`C:\...`)

Không khuyến nghị, nhưng vẫn làm được:

```bash
cd /mnt/c/Users/<user>/path/to/car-damage-mlops
claude
```

Nhược điểm: I/O chậm, lỗi quyền file / xuống dòng CRLF, `.venv` tạo bên Linux không dùng được từ Windows. Nên chuyển hẳn sang WSL:

```bash
cd ~/projects && git clone https://github.com/daisyngocha/car-damage-mlops.git
```

Muốn sửa bằng editor: cài VS Code + extension **WSL**, rồi trong terminal WSL gõ `code .`. Có thể cài thêm extension **Claude Code** cho VS Code để dùng Claude ngay trong editor (vẫn chạy trong WSL).

---

## 4. Trường hợp: điều khiển phiên WSL từ điện thoại / web (Remote Control)

Dùng khi muốn rời máy nhưng vẫn để Claude làm việc trên chính thư mục WSL (data, GPU, `.venv` local).

```bash
cd ~/projects/car-damage-mlops
claude remote-control        # hoặc gõ /remote-control trong phiên đang mở
```

Phiên sẽ hiện trong app Claude (mục Code) trên web/điện thoại. Prompt từ app, Claude vẫn chạy trên máy mình.

Điều kiện: máy bật, không sleep, terminal đó vẫn mở.

---

## 5. Trường hợp: Cloud session (claude.ai/code hoặc app Claude)

Claude chạy trong container trên cloud, clone repo từ GitHub. Thường làm trên branch `claude/...`.

Hợp với:

- Refactor, viết test, sửa README, review PR, sửa CI.
- Giao việc rồi đi làm việc khác, không cần mở máy.

Không hợp với:

- Train model thật, DVC pull data, W&B với data local, cần GPU → dùng CLI trong WSL.

Lấy kết quả từ cloud về máy:

```bash
git fetch origin
git checkout claude/<tên-branch>      # xem thử
# hoặc: tạo PR từ branch đó → merge → git checkout main && git pull
```

Cloud session không thấy file chưa commit trong WSL. Muốn nó thấy thì commit + push trước.

---

## 6. Chọn công cụ nào?

| Việc | Dùng |
|---|---|
| Train, evaluate, DVC, W&B, Docker build local | CLI trong WSL |
| Như trên nhưng đang ở ngoài, cầm điện thoại | Remote Control |
| Sửa code / test / docs thuần, không cần data | Cloud session hoặc CLI đều được |
| Review PR, sửa CI đỏ | Cloud session |
| Đọc / sửa code bằng mắt | VS Code (WSL) + extension Claude Code |

---

## 7. An toàn và những điều không làm

- Không commit `data/`, `models/`, `*.pth.tar`, `.env`, `wandb/` (đã có trong `.gitignore`). Data đi qua DVC, không qua git.
- Không dán API key, token, mật khẩu vào prompt. Để trong `.env`, chỉ nói Claude "đọc biến môi trường X".
- Trước khi cho Claude chạy lệnh xoá (`rm -rf`, `git reset --hard`, `git push --force`) phải đọc kỹ.
- Luôn xem `git diff` trước khi commit, kể cả khi Claude đã tự kiểm tra.
- Thêm thư viện: `uv add <pkg>` (hoặc `uv add --dev <pkg>`), commit cả `pyproject.toml` và `uv.lock`.

---

## 8. Xử lý sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `claude: command not found` | Chưa cài, hoặc PATH chưa nhận: mở terminal mới, hoặc `source ~/.bashrc` |
| Trình duyệt không mở khi login | Nhấn `c` copy URL, mở thủ công bên Windows |
| Login báo lỗi / code hết hạn | Thoát, chạy lại `claude` hoặc `/login` để lấy URL mới |
| Bị proxy công ty chặn | Thử mạng khác hoặc nhờ IT mở `claude.com`, `platform.claude.com`, `api.anthropic.com` |
| `git push` hỏi mật khẩu / bị từ chối | Chưa xác thực GitHub trong WSL: `gh auth login` hoặc SSH key (mục 2.6) |
| `uv: command not found` | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, mở terminal mới |
| Test pass ở local nhưng CI đỏ | Quên commit `uv.lock`, hoặc test phụ thuộc data local. Đọc log job trên tab Actions |
| Claude "không thấy" file mình vừa sửa | Kiểm tra mình đang ở môi trường nào (mục 1). Cloud session cần push trước |
| Hết hạn mức gói Pro | `/usage` để xem. Đợi reset, dùng `/clear` thường xuyên để tiết kiệm |
| Claude làm sai hướng | `Esc` để dừng, `Esc Esc` để quay lại prompt trước, viết lại rõ hơn |

---

## 9. Checklist nhanh mỗi buổi làm

```
[ ] cd ~/projects/car-damage-mlops && git checkout main && git pull
[ ] uv sync --locked
[ ] git checkout -b feat/<việc>
[ ] claude   (chọn mode bằng Shift+Tab)
[ ] làm việc → ! git diff → ! uv run pytest -q
[ ] commit + push → tạo PR → CI xanh → merge
[ ] git checkout main && git pull
```

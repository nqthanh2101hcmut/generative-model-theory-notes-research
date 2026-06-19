---
title: "Technical Report 04 - Diffusion and Score-Based Generative Models"
subtitle: "SDE, Score Matching, Reverse Dynamics and Sampling Error"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# 1. Ý tưởng cốt lõi của diffusion models

Diffusion models xây dựng hai quá trình:

1. **Forward process:** biến dữ liệu thành noise.
2. **Reverse process:** biến noise trở lại dữ liệu.

Nếu \(X_0\sim p_{\mathrm{data}}\), forward process tạo \(X_t\) ngày càng nhiễu. Khi \(t=T\), \(X_T\) gần Gaussian. Sau đó ta học reverse process để lấy mẫu từ data distribution.

# 2. Forward SDE

Dạng tổng quát:

\[
dX_t=f(X_t,t)dt+g(t)dW_t.
\]

Trong đó:

- \(f\) là drift;
- \(g\) là diffusion coefficient;
- \(W_t\) là Brownian motion.

Một ví dụ phổ biến là Ornstein-Uhlenbeck/VP-type diffusion:

\[
dX_t=-\frac12\beta(t)X_tdt+\sqrt{\beta(t)}dW_t.
\]

Forward process được thiết kế sao cho phân phối cuối gần Gaussian chuẩn.

# 3. Fokker-Planck equation

Nếu \(p_t\) là mật độ của \(X_t\), thì \(p_t\) thỏa Fokker-Planck equation:

\[
\partial_t p_t(x)
=
-\nabla\cdot(f(x,t)p_t(x))
+
\frac12 g(t)^2\Delta p_t(x).
\]

Đây là cầu nối giữa SDE và PDE. Trong nghiên cứu lý thuyết, ta thường phân tích diffusion qua cả hai góc nhìn.

# 4. Reverse-time SDE

Kết quả nền tảng trong score-based modeling là reverse process có dạng:

\[
dX_t=
\left[f(X_t,t)-g(t)^2\nabla_x\log p_t(X_t)\right]dt
+g(t)d\overline W_t,
\]

khi chạy thời gian ngược. Thành phần quan trọng là score:

\[
s_t(x)=\nabla_x\log p_t(x).
\]

Do đó nếu học được score \(s_t\), ta có thể mô phỏng reverse SDE để sinh dữ liệu.

# 5. Score matching

Score matching học một model \(s_\theta(x,t)\) sao cho:

\[
s_\theta(x,t)\approx \nabla_x\log p_t(x).
\]

Loss lý tưởng:

\[
\mathcal L(\theta)
=
\mathbb E_{t,x\sim p_t}
\left[
\|s_\theta(x,t)-\nabla\log p_t(x)\|^2
\right].
\]

Vấn đề là score thật \(\nabla\log p_t\) không biết. Denoising score matching giải quyết bằng cách dùng conditional noising distribution.

# 6. Denoising score matching

Giả sử:

\[
x_t=\alpha_t x_0+\sigma_t\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I).
\]

Conditional score là:

\[
\nabla_{x_t}\log p(x_t|x_0)
=
-\frac{x_t-\alpha_t x_0}{\sigma_t^2}.
\]

Training loss có dạng:

\[
\mathbb E_{t,x_0,\varepsilon}
\left[
\left\|s_\theta(x_t,t)+\frac{x_t-\alpha_t x_0}{\sigma_t^2}\right\|^2
\right].
\]

Đây là công thức thực hành cốt lõi của score-based training.

# 7. Probability flow ODE

Ngoài reverse SDE, có một ODE sinh cùng marginal distributions:

\[
\frac{dX_t}{dt}
=
f(X_t,t)-\frac12g(t)^2\nabla\log p_t(X_t).
\]

ODE này gọi là probability flow ODE. Nó cho phép lấy mẫu deterministic và kết nối diffusion với continuous normalizing flows.

# 8. Sampling error

Trong thực tế, reverse SDE/ODE được rời rạc hóa. Nếu dùng Euler method với step size \(h\), có sai số:

\[
\varepsilon_{\mathrm{disc}}=\text{error due to time discretization}.
\]

Nếu score học sai:

\[
\varepsilon_{\mathrm{score}}
=
\int_0^T\mathbb E\|s_\theta(X_t,t)-s_t(X_t)\|^2dt,
\]

thì sai số này lan truyền qua dynamics. Câu hỏi nghiên cứu:

\[
\varepsilon_{\mathrm{score}} + \varepsilon_{\mathrm{disc}}
\quad\Rightarrow\quad
W_p(\rho_\theta,\mu) \text{ nhỏ như thế nào?}
\]

# 9. Optimization view

Training score model là bài toán:

\[
\min_\theta \mathcal L_{\mathrm{DSM}}(\theta).
\]

Nhưng nếu nhìn ở mức phân phối sau sampling, ta có thể đặt bài toán:

\[
\min_{\rho}
\mathbb E_\rho[\ell(x)]
+
\lambda\mathrm{KL}(\rho\|\rho_0).
\]

Đây là distributional optimization view of diffusion alignment. Nó không thay thế score training, nhưng cung cấp một tầng lý thuyết khác cho alignment.

# 10. Diffusion alignment

Giả sử \(\rho_0\) là phân phối base diffusion model. Ta muốn tìm phân phối mới \(\rho\) tối ưu reward/cost nhưng vẫn gần \(\rho_0\):

\[
\min_\rho
\mathbb E_\rho[\ell(x)]
+
\lambda\mathrm{KL}(\rho\|\rho_0).
\]

Nghiệm dạng formal:

\[
\frac{d\rho^\star}{d\rho_0}(x)
\propto
\exp(-\ell(x)/\lambda).
\]

Câu hỏi khó: làm sao sample từ \(\rho^\star\) bằng cách điều chỉnh score/reverse dynamics?

# 11. Các giả định lý thuyết thường gặp

Trong phân tích diffusion, paper thường giả định:

- score Lipschitz;
- score error bị chặn;
- data distribution có moment hữu hạn;
- density trơn hoặc thuộc Sobolev/Besov space;
- drift/diffusion thỏa điều kiện tồn tại nghiệm SDE;
- time discretization đủ nhỏ.

Một hướng nghiên cứu là làm yếu các giả định này.

# 12. Research questions

1. Nếu score error nhỏ trong \(L^2(p_t)\), distribution error nhỏ theo metric nào?
2. Nếu data nằm gần manifold thấp chiều, diffusion có rate phụ thuộc intrinsic dimension không?
3. Nếu score không trơn, reverse SDE còn ổn định không?
4. Nếu thêm primal-dual alignment, constraint violation giảm thế nào?
5. Nếu thay KL bằng OT hoặc thêm OT regularization, sampling dynamics thay đổi ra sao?
6. Có thể chứng minh variational stability cho diffusion alignment objective không?

# 13. Mini-project diffusion 2D

Dataset:

- Gaussian mixture;
- two moons;
- checkerboard;
- spiral.

Các bước:

1. Tạo forward noising process.
2. Huấn luyện MLP score network.
3. Lấy mẫu bằng reverse Euler-Maruyama.
4. Vẽ score field.
5. Đo MMD hoặc sliced Wasserstein.
6. Thay đổi noise schedule và quan sát stability.

# 14. Bài tập tự kiểm tra

1. Derive conditional score của Gaussian noising.
2. Viết Fokker-Planck equation cho SDE đơn giản.
3. Giải thích tại sao reverse SDE cần score.
4. So sánh reverse SDE và probability flow ODE.
5. Code Ornstein-Uhlenbeck process.
6. Mô phỏng Euler-Maruyama và quan sát sai số khi step size thay đổi.
7. Viết error decomposition cho diffusion sampling.

# 15. Kết luận

Diffusion models là nơi probability, SDE, PDE, score matching và optimization gặp nhau. Nếu muốn làm theoretical foundations, không nên chỉ nhớ công thức training loss. Cần hiểu score là object toán học, reverse dynamics phụ thuộc vào score, và sampling error là kết quả của nhiều tầng sai số: statistical, optimization, approximation và discretization.

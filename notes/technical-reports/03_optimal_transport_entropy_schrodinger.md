---
title: "Technical Report 03 - Optimal Transport, Entropy and Schrödinger Bridges"
subtitle: "A Geometric and Variational View of Generative Modeling"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# 1. Vì sao optimal transport quan trọng?

Generative modeling có thể được hiểu là bài toán vận chuyển một phân phối dễ lấy mẫu thành phân phối dữ liệu thật:

\[
p_0 \longrightarrow \mu.
\]

Nếu \(p_0\) là Gaussian noise và \(\mu\) là data distribution, thì ta cần một cơ chế vận chuyển khối lượng xác suất. Optimal transport cung cấp ngôn ngữ hình học cho câu hỏi này.

# 2. Couplings

Cho hai phân phối \(\mu\) và \(\nu\), một coupling là một phân phối \(\pi\) trên \(\mathcal X\times\mathcal Y\) sao cho marginal thứ nhất là \(\mu\) và marginal thứ hai là \(\nu\):

\[
\pi(A\times\mathcal Y)=\mu(A),
\qquad
\pi(\mathcal X\times B)=\nu(B).
\]

Tập các coupling ký hiệu:

\[
\Pi(\mu,\nu).
\]

Trong flow matching, coupling xác định cách ghép noise sample \(x_0\) với data sample \(x_1\).

# 3. Kantorovich problem

Bài toán optimal transport dạng Kantorovich:

\[
\mathrm{OT}_c(\mu,\nu)
=
\inf_{\pi\in\Pi(\mu,\nu)}
\int c(x,y)d\pi(x,y).
\]

Nếu \(c(x,y)=\|x-y\|^p\), ta thu được Wasserstein distance:

\[
W_p(\mu,\nu)
=
\left(
\inf_{\pi\in\Pi(\mu,\nu)}
\int \|x-y\|^p d\pi(x,y)
\right)^{1/p}.
\]

# 4. Dual formulation

Với chi phí \(c\), dual của OT có dạng:

\[
\sup_{\varphi,\psi}
\left\{
\int\varphi d\mu+\int\psi d\nu:
\varphi(x)+\psi(y)\le c(x,y)
\right\}.
\]

Dual formulation rất quan trọng trong analysis vì nó biến bài toán trên couplings thành bài toán trên potentials.

# 5. Entropic optimal transport

OT cổ điển có thể không trơn và tốn tính toán. Entropic OT thêm entropy regularization:

\[
\mathrm{OT}_\varepsilon(\mu,\nu)
=
\inf_{\pi\in\Pi(\mu,\nu)}
\left\{
\int c(x,y)d\pi(x,y)
+
\varepsilon\mathrm{KL}(\pi\|\mu\otimes\nu)
\right\}.
\]

Vai trò của \(\varepsilon\):

- \(\varepsilon>0\) làm bài toán trơn hơn;
- giúp dùng Sinkhorn algorithm;
- tạo bias so với OT gốc;
- khi \(\varepsilon\to 0\), kỳ vọng hội tụ về OT cổ điển.

# 6. Sinkhorn algorithm

Trong trường hợp rời rạc, có cost matrix \(C\), marginals \(a,b\), kernel:

\[
K_{ij}=\exp(-C_{ij}/\varepsilon).
\]

Transport plan có dạng:

\[
P=\mathrm{diag}(u)K\mathrm{diag}(v),
\]

trong đó \(u,v\) được cập nhật:

\[
u \leftarrow \frac{a}{Kv},
\qquad
v \leftarrow \frac{b}{K^Tu}.
\]

Sinkhorn là công cụ tính toán nền tảng cho entropic OT, OT-CFM và Schrödinger bridge rời rạc.

# 7. Dynamic OT và Benamou-Brenier

Với \(W_2\), ta có formulation động:

\[
W_2^2(\mu_0,\mu_1)
=
\inf_{(\rho_t,v_t)}
\int_0^1\int \|v_t(x)\|^2 d\rho_t(x)dt,
\]

subject to continuity equation:

\[
\partial_t\rho_t+\nabla\cdot(\rho_t v_t)=0,
\qquad
\rho_0=\mu_0,
\quad
\rho_1=\mu_1.
\]

Đây là cầu nối trực tiếp với flow matching, vì flow matching cũng học vector field \(v_t\).

# 8. OT cho flow matching

Trong conditional flow matching, ta lấy cặp \((x_0,x_1)\) và định nghĩa path:

\[
x_t=(1-t)x_0+tx_1.
\]

Velocity tương ứng:

\[
u_t=x_1-x_0.
\]

Nếu \((x_0,x_1)\) được ghép ngẫu nhiên độc lập, trajectory có thể dài và khó học. Nếu ghép bằng OT, pairing ngắn hơn và flow thường đơn giản hơn.

Bài toán nghiên cứu:

\[
\text{OT pairing improves training stability, but what is the variational limit of minibatch OT?}
\]

# 9. Minibatch OT bias

Trong thực tế không tính OT trên toàn bộ dataset mà trên minibatch. Gọi objective đầy đủ là \(F\), objective minibatch là \(F_B\). Câu hỏi:

\[
F_B \to F ?
\]

Nếu có hội tụ, theo nghĩa nào?

- pointwise convergence;
- convergence in expectation;
- Gamma convergence;
- convergence of minimizers.

Đây là một hướng research rất phù hợp với variational analysis.

# 10. Schrödinger bridge

Schrödinger bridge là bài toán tìm process \(\mathbb P\) gần một reference process \(\mathbb Q\), sao cho hai biên cố định:

\[
X_0\sim\mu_0,
\qquad
X_1\sim\mu_1.
\]

Bài toán:

\[
\min_{\mathbb P}
\mathrm{KL}(\mathbb P\|\mathbb Q)
\quad
\text{subject to}\quad
\mathbb P_0=\mu_0,
\;\mathbb P_1=\mu_1.
\]

Nếu \(\mathbb Q\) là Brownian motion, Schrödinger bridge là phiên bản entropy-regularized của dynamic OT.

# 11. Liên hệ giữa OT, Schrödinger bridge và diffusion

| Framework | Objective | Dynamics | Regularization |
|---|---|---|---|
| OT | minimize transport cost | deterministic transport | none |
| Entropic OT | cost + entropy | soft coupling | entropy |
| Schrödinger bridge | path-space KL | stochastic process | KL to reference process |
| Diffusion model | reverse noising process | SDE | score/denoising loss |
| Flow matching | vector field regression | ODE | path design |

Schrödinger bridge là điểm giao giữa stochastic processes, optimal transport và entropy.

# 12. Variational questions

Các câu hỏi toán học quan trọng:

1. Khi \(\varepsilon\to 0\), \(\mathrm{OT}_\varepsilon\) hội tụ về OT cổ điển như thế nào?
2. Nghiệm entropic OT có ổn định khi \(\mu_n\to\mu\) không?
3. Minibatch OT objective có Gamma-converges về objective đầy đủ không?
4. Schrödinger bridge có KKT/duality trên path space như thế nào?
5. Flow matching dùng OT path có stability tốt hơn random path không?
6. Có thể chặn sai số transport plan khi cost hoặc marginals bị nhiễu không?

# 13. Technical theorem template

**Theorem template.** Giả sử \(\mu_n\Rightarrow\mu\), \(\nu_n\Rightarrow\nu\), các moments bậc \(p\) hội tụ, và cost \(c(x,y)=\|x-y\|^p\). Khi đó:

\[
W_p(\mu_n,\nu_n)\to W_p(\mu,\nu).
\]

Một phiên bản mạnh hơn có thể dùng để chứng minh stability của OT-regularized distributional optimization.

# 14. Mini-project đề xuất

1. Code Sinkhorn từ đầu.
2. Tính transport plan giữa hai Gaussian mixtures.
3. So sánh random pairing và OT pairing trong flow matching 2D.
4. Thử các giá trị \(\varepsilon\) và quan sát bias/stability.
5. Vẽ heatmap của coupling.
6. Đo độ dài trung bình của trajectory.

# 15. Bài tập tự kiểm tra

1. Viết định nghĩa coupling và kiểm tra marginal constraints.
2. Chứng minh \(W_p\) là metric trong trường hợp đơn giản.
3. Giải thích tại sao entropic OT trơn hơn OT cổ điển.
4. Derive Sinkhorn scaling form \(P=\mathrm{diag}(u)K\mathrm{diag}(v)\).
5. Giải thích Benamou-Brenier liên hệ với flow matching.
6. Viết bài toán Schrödinger bridge và so sánh với entropic OT.

# 16. Kết luận

Optimal transport cung cấp hình học cho generative modeling. Entropic regularization cung cấp stability và computational tractability. Schrödinger bridge cung cấp path-space view kết nối diffusion, stochastic control và optimal transport. Đây là một trong những nhánh toán quan trọng nhất nếu muốn nghiên cứu flow matching và generative model theory theo hướng Toán ứng dụng.

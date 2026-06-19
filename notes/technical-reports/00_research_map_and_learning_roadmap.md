---
title: "Technical Report 00 - Research Map and Learning Roadmap"
subtitle: "Optimization, Probability, Statistics and Generative Model Theory"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# 0. Mục tiêu của bộ ghi chú

Bộ ghi chú này được thiết kế như một **research notebook có cấu trúc học thuật** cho hướng:

\[
\text{Optimization + Probability + Statistics + Optimal Transport}
\quad \longrightarrow \quad
\text{Diffusion/Flow Generative Model Theory}.
\]

Mục tiêu không phải là học deep learning theo kiểu dùng thư viện và chạy mô hình lớn. Mục tiêu là xây dựng một nền toán đủ vững để đặt và giải các câu hỏi nghiên cứu về:

1. tối ưu trên không gian phân phối xác suất;
2. bài toán có chính quy entropy/KL;
3. điều kiện tối ưu, đối ngẫu và thuật toán primal-dual;
4. hội tụ biến phân và ổn định nghiệm;
5. sai số thống kê do dữ liệu hữu hạn;
6. sai số sampling do giải SDE/ODE;
7. ứng dụng vào diffusion models, score-based models, flow matching và Schrödinger bridges.

Điểm nhìn trung tâm của repo là:

\[
\boxed{
\text{Generative modeling is distributional optimization.}
}
\]

Một mô hình sinh không chỉ tạo ra từng điểm dữ liệu riêng lẻ. Nó tạo ra một phân phối \(\rho_\theta\), và mục tiêu là làm cho phân phối đó gần với phân phối dữ liệu thật \(\mu\):

\[
\rho_\theta \approx \mu.
\]

Do đó, nền toán của generative models bao gồm measure theory, probability, statistics, optimization, optimal transport, stochastic processes, ODE/PDE và approximation theory.

# 1. Đối tượng toán học trung tâm

Trong hướng này, ta thường xét một bài toán tối ưu trên không gian phân phối xác suất:

\[
\min_{\rho \in \mathcal P(\mathcal X)}
F(\rho)
=
\mathbb E_{x\sim \rho}[\ell(x)]
+
\lambda \mathrm{KL}(\rho\|\rho_0)
+
\gamma \mathrm{OT}_\varepsilon(\rho,\mu),
\]

kèm theo các ràng buộc dạng kỳ vọng:

\[
\mathbb E_{x\sim \rho}[c_j(x)] \le b_j,
\qquad j=1,\dots,m.
\]

Trong đó:

| Ký hiệu | Ý nghĩa |
|---|---|
| \(\rho\) | phân phối cần tối ưu, ví dụ phân phối sinh ra sau alignment |
| \(\rho_0\) | phân phối tham chiếu, ví dụ base generative model |
| \(\mu\) | phân phối dữ liệu thật hoặc phân phối mục tiêu |
| \(\ell\) | loss/reward/cost |
| \(\mathrm{KL}\) | khoảng cách entropy tương đối, giữ \(\rho\) không lệch quá xa \(\rho_0\) |
| \(\mathrm{OT}_\varepsilon\) | optimal transport có chính quy entropy |
| \(c_j\) | các ràng buộc về safety, diversity, quality, cost hoặc physics |

Đây là một khung rất linh hoạt: nếu bỏ \(\mathrm{OT}\), ta có KL-regularized distributional optimization; nếu bỏ KL và dùng OT, ta gần với flow matching và Wasserstein geometry; nếu xét bài toán trên path space, ta đi đến Schrödinger bridge.

# 2. Bản đồ các nhánh toán

## 2.1. Probability and measure theory

Generative model là bài toán về phân phối. Vì vậy ta cần:

- probability measures;
- measurable spaces;
- pushforward measures;
- weak convergence;
- empirical measures;
- conditional distributions;
- Radon-Nikodym derivatives.

Ví dụ, nếu \(Z\sim p_0\) là noise và \(T_\theta\) là map sinh dữ liệu, thì phân phối sinh ra là pushforward:

\[
\rho_\theta = (T_\theta)_\# p_0.
\]

## 2.2. Statistics

Thực tế không biết \(\mu\), chỉ có mẫu:

\[
x_1,\dots,x_n \sim \mu.
\]

Ta thay \(\mu\) bằng empirical measure:

\[
\mu_n = \frac1n \sum_{i=1}^n \delta_{x_i}.
\]

Câu hỏi thống kê là:

\[
\text{Nghiệm của } F_n \text{ có hội tụ về nghiệm của } F \text{ không?}
\]

## 2.3. Optimization and variational analysis

Training hoặc alignment thường là một bài toán tối ưu. Khi có ràng buộc, ta cần Lagrangian:

\[
\mathcal L(\rho,\alpha)
=
F(\rho)+\sum_{j=1}^m \alpha_j
\bigl(\mathbb E_\rho[c_j]-b_j\bigr),
\qquad \alpha_j\ge 0.
\]

Cần nghiên cứu:

- existence of minimizers;
- KKT conditions;
- strong duality;
- primal-dual gap;
- stability of minimizers;
- Gamma/epi/variational convergence.

## 2.4. Optimal transport

Optimal transport đo cách vận chuyển một phân phối thành phân phối khác:

\[
\mathrm{OT}_c(\mu,\nu)
=
\inf_{\pi\in \Pi(\mu,\nu)} \int c(x,y)\,d\pi(x,y).
\]

Trong flow matching, OT giúp chọn coupling \(\pi(x_0,x_1)\) tốt giữa noise và data, từ đó tạo trajectory dễ học hơn.

## 2.5. SDE/PDE/ODE

Diffusion models dùng SDE:

\[
dX_t = f(X_t,t)dt + g(t)dW_t.
\]

Flow matching dùng ODE:

\[
\frac{dX_t}{dt}=v_t(X_t).
\]

Mật độ xác suất tương ứng thỏa PDE như Fokker-Planck hoặc continuity equation.

# 3. Bản đồ mô hình sinh

| Mô hình | Đối tượng học | Công cụ toán chính |
|---|---|---|
| Score-based diffusion | score \(\nabla_x\log p_t(x)\) | SDE, Fisher divergence, score matching |
| DDPM/SDE diffusion | reverse denoising process | Markov chains, reverse SDE, Fokker-Planck |
| Flow matching | vector field \(v_t\) | ODE, continuity equation, OT |
| Schrödinger bridge | stochastic process nối hai biên | entropy, stochastic control, path-space KL |
| Distributional alignment | phân phối tối ưu \(\rho^\star\) | KL, dual averaging, primal-dual methods |

# 4. Kiến trúc luận án khả thi

Một luận án tốt trong hướng này có thể chia thành ba trục.

## Trục A - Variational foundations

Bài toán:

\[
\min_{\rho\in \mathcal P(\mathcal X)} F(\rho).
\]

Nghiên cứu:

1. điều kiện tồn tại nghiệm;
2. lower semicontinuity và coercivity;
3. KKT/duality;
4. ổn định nghiệm khi \(\mu\) được thay bởi \(\mu_n\);
5. hội tụ biến phân của các bài toán xấp xỉ.

## Trục B - Algorithms

Xây dựng thuật toán primal-dual:

\[
\alpha_{k+1}
=
\bigl[\alpha_k+\eta_k(\mathbb E_{\rho_k}[c]-b)\bigr]_+,
\]

và primal update dạng mirror descent/KL proximal:

\[
\rho_{k+1}
=
\arg\min_\rho
\left\{
\langle G_k,\rho\rangle
+
\frac1{\eta_k}\mathrm{KL}(\rho\|\rho_k)
+\gamma \mathrm{OT}_\varepsilon(\rho,\mu_n)
\right\}.
\]

## Trục C - Generative model applications

Áp dụng vào:

- diffusion alignment;
- flow matching with OT paths;
- constrained generation;
- toy 2D distributions;
- error decomposition: optimization + statistics + approximation + sampling.

# 5. Lộ trình tự học 12 tháng

| Tháng | Chủ đề | Output bắt buộc |
|---|---|---|
| 1 | Linear algebra, real analysis, Python | notebook gradient descent |
| 2 | Probability basics, empirical distribution | note về pushforward và empirical measures |
| 3 | Convex optimization | proof KKT đơn giản |
| 4 | KL, entropy, mirror descent | code KL optimization trên simplex |
| 5 | Measure-theoretic probability | note về weak convergence |
| 6 | Statistical learning theory | error decomposition note |
| 7 | Variational analysis | theorem về stability of minimizers |
| 8 | Optimal transport | code Sinkhorn và transport heatmap |
| 9 | SDE and diffusion | code toy score matching |
| 10 | ODE and flow matching | code flow matching toy |
| 11 | Paper reading | summary 10 papers |
| 12 | Research proposal | technical report 20-30 pages |

# 6. Cách đọc paper

Mỗi paper nên được đọc theo 5 lớp:

1. **Problem.** Paper giải câu hỏi nào?
2. **Mathematical object.** Đối tượng chính là distribution, score, vector field, coupling hay path measure?
3. **Assumptions.** Giả định smoothness, compactness, boundedness, log-concavity, finite moments là gì?
4. **Main theorem.** Kết quả được đo bằng KL, TV, Wasserstein, Fisher divergence hay optimization gap?
5. **Limitation.** Paper chưa xử lý được trường hợp nào?

Mẫu ghi chú nên có:

```text
Paper:
Central question:
Mathematical setting:
Key assumptions:
Main theorem:
Proof skeleton:
Connection to my research:
Possible extension:
```

# 7. Research questions ban đầu

Một số câu hỏi có thể phát triển thành đề tài:

1. Khi nào KL-regularized distributional optimization có nghiệm duy nhất?
2. Khi reward/cost bị nhiễu, nghiệm \(\rho^\star\) thay đổi liên tục theo metric nào?
3. Nếu thay \(\mu\) bằng \(\mu_n\), nghiệm empirical có hội tụ về nghiệm population không?
4. Minibatch entropic OT có hội tụ biến phân về entropic OT đầy đủ không?
5. Sai số vector field trong flow matching lan truyền qua ODE như thế nào?
6. Sai số score trong diffusion lan truyền qua reverse SDE như thế nào?
7. Có thể thiết kế primal-dual algorithm cho constrained diffusion alignment không?
8. Constraint violation giảm theo tốc độ nào?
9. Entropy regularization có cải thiện stability nhưng tạo bias như thế nào?
10. Có thể tách tổng sai số thành optimization + statistical + sampling + approximation không?

# 8. Chuẩn đầu ra cá nhân

Sau khi hoàn thành repo này, người học nên có:

- 1 bản research proposal 8-10 trang;
- 1 technical report nền tảng 30-50 trang;
- 3 toy projects: KL optimization, Sinkhorn OT, flow/score matching;
- 10 paper summaries;
- 1 theorem nhỏ có proof đầy đủ;
- 1 GitHub repo gọn, có README và reproducible code.

# 9. Tài liệu nền tảng đề xuất

- Boyd and Vandenberghe, *Convex Optimization*.
- Rockafellar and Wets, *Variational Analysis*.
- Villani, *Optimal Transport: Old and New*.
- Ambrosio, Gigli and Savaré, *Gradient Flows*.
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*.
- Lipman et al., *Flow Matching for Generative Modeling*.
- Tong et al., *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*.
- Tong et al., *Simulation-Free Schrödinger Bridges via Score and Flow Matching*.
- Kawata, Oko, Nitanda and Suzuki, *Direct Distributional Optimization for Provable Alignment of Diffusion Models*.

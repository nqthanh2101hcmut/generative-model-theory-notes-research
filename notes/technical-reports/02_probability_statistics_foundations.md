---
title: "Technical Report 02 - Probability and Statistics Foundations"
subtitle: "Measures, Empirical Distributions and Error Decomposition for Generative Models"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# 1. Tại sao probability là nền tảng của generative models?

Một generative model không chỉ là một neural network. Nó là một cơ chế sinh mẫu từ một phân phối. Nếu \(Z\sim p_0\) là nhiễu Gaussian và \(G_\theta\) là generator, thì mẫu sinh là:

\[
X=G_\theta(Z).
\]

Phân phối của \(X\) được ký hiệu:

\[
\rho_\theta=(G_\theta)_\#p_0.
\]

Mục tiêu là:

\[
\rho_\theta\approx \mu,
\]

trong đó \(\mu\) là phân phối dữ liệu thật. Do đó các khái niệm xác suất như probability measure, pushforward, weak convergence, Wasserstein distance và empirical measure là bắt buộc.

# 2. Probability space và random variables

Một không gian xác suất là bộ ba:

\[
(\Omega,\mathcal F,\mathbb P),
\]

trong đó:

- \(\Omega\) là không gian mẫu;
- \(\mathcal F\) là sigma-algebra các biến cố;
- \(\mathbb P\) là độ đo xác suất.

Một random variable là ánh xạ đo được:

\[
X:\Omega\to\mathcal X.
\]

Phân phối của \(X\) là pushforward measure:

\[
\mathcal L(X)=X_\#\mathbb P,
\]

được định nghĩa bởi:

\[
X_\#\mathbb P(A)=\mathbb P(X\in A).
\]

Trong generative modeling, generator chính là một random variable phụ thuộc tham số.

# 3. Pushforward measure

Cho ánh xạ đo được \(T:\mathcal X\to\mathcal Y\) và measure \(\mu\) trên \(\mathcal X\), pushforward \(T_\#\mu\) được định nghĩa bởi:

\[
(T_\#\mu)(B)=\mu(T^{-1}(B)).
\]

Nếu \(Z\sim p_0\) và \(X=T(Z)\), thì:

\[
\mathcal L(X)=T_\#p_0.
\]

Đây là công thức nền cho normalizing flows và flow-based generative models.

# 4. Empirical measure

Trong thực tế ta không biết \(\mu\). Ta chỉ có dữ liệu:

\[
x_1,\dots,x_n\sim\mu.
\]

Empirical measure là:

\[
\mu_n=\frac1n\sum_{i=1}^n\delta_{x_i}.
\]

Với hàm test \(\varphi\), ta có:

\[
\int \varphi(x)d\mu_n(x)
=
\frac1n\sum_{i=1}^n \varphi(x_i).
\]

Luật số lớn nói rằng với \(\varphi\) đủ tốt:

\[
\int \varphi d\mu_n \to \int \varphi d\mu.
\]

Đây là nền cho empirical risk minimization.

# 5. Các kiểu hội tụ của phân phối

## 5.1. Weak convergence

Ta nói \(\mu_n\Rightarrow\mu\) nếu với mọi hàm liên tục bị chặn \(\varphi\):

\[
\int \varphi d\mu_n \to \int \varphi d\mu.
\]

Weak convergence là ngôn ngữ tự nhiên khi nghiên cứu empirical measures và stability of distributions.

## 5.2. Total variation

\[
\|\mu-\nu\|_{\mathrm{TV}}
=
\sup_A |\mu(A)-\nu(A)|.
\]

TV rất mạnh nhưng đôi khi quá khắt khe, đặc biệt khi hai phân phối có support không trùng.

## 5.3. Wasserstein distance

Với \(p\ge 1\):

\[
W_p(\mu,\nu)
=
\left(
\inf_{\pi\in\Pi(\mu,\nu)}
\int \|x-y\|^p d\pi(x,y)
\right)^{1/p}.
\]

Wasserstein phù hợp với generative models vì nó đo chi phí vận chuyển khối lượng, vẫn có ý nghĩa khi supports không trùng.

# 6. Score function

Nếu \(p(x)\) là mật độ trơn, score function là:

\[
s_p(x)=\nabla_x\log p(x).
\]

Score chỉ ra hướng tăng nhanh nhất của log-density. Score-based generative models học \(s_{p_t}(x)\) cho các mức nhiễu \(t\).

Một tính chất quan trọng:

\[
\nabla_x p(x)=p(x)\nabla_x\log p(x).
\]

Do đó score xuất hiện tự nhiên trong reverse diffusion SDE.

# 7. Statistical learning view

Trong population setting, ta muốn tối ưu:

\[
R(\theta)=\mathbb E_{X\sim\mu}[L(\theta,X)].
\]

Nhưng thực tế dùng empirical risk:

\[
\widehat R_n(\theta)=\frac1n\sum_{i=1}^n L(\theta,x_i).
\]

Sai số tổng quát có thể tách:

\[
R(\widehat\theta)-\inf_{\theta\in\Theta}R(\theta)
=
\underbrace{R(\widehat\theta)-\widehat R_n(\widehat\theta)}_{\text{generalization}}
+
\underbrace{\widehat R_n(\widehat\theta)-\inf_\theta \widehat R_n(\theta)}_{\text{optimization}}
+
\underbrace{\inf_\theta\widehat R_n(\theta)-\inf_\theta R(\theta)}_{\text{statistical fluctuation}}.
\]

Trong diffusion/flow theory, decomposition thường có thêm approximation error và sampling error.

# 8. Error decomposition cho generative models

Một mục tiêu lý thuyết điển hình:

\[
d(\rho_{\widehat\theta},\mu)
\le
\varepsilon_{\mathrm{stat}}
+
\varepsilon_{\mathrm{opt}}
+
\varepsilon_{\mathrm{approx}}
+
\varepsilon_{\mathrm{disc}}
+
\varepsilon_{\mathrm{sample}}.
\]

Trong đó:

| Thành phần | Ý nghĩa |
|---|---|
| \(\varepsilon_{\mathrm{stat}}\) | sai số do dữ liệu hữu hạn |
| \(\varepsilon_{\mathrm{opt}}\) | sai số do thuật toán tối ưu chưa hội tụ |
| \(\varepsilon_{\mathrm{approx}}\) | sai số do neural network không biểu diễn được score/vector field thật |
| \(\varepsilon_{\mathrm{disc}}\) | sai số rời rạc hóa ODE/SDE |
| \(\varepsilon_{\mathrm{sample}}\) | sai số do quy trình lấy mẫu |

Đây là framework rất quan trọng nếu muốn đi theo hướng Research Scientist về theoretical foundations.

# 9. Stability under empirical approximation

Xét bài toán population:

\[
\rho^\star\in\arg\min_{\rho}F(\rho;\mu),
\]

và bài toán empirical:

\[
\rho_n^\star\in\arg\min_{\rho}F(\rho;\mu_n).
\]

Câu hỏi:

\[
\rho_n^\star\to \rho^\star ?
\]

Một hướng chứng minh:

1. chứng minh \(\mu_n\Rightarrow\mu\);
2. chứng minh \(F(\cdot;\mu_n)\) Gamma-converges tới \(F(\cdot;\mu)\);
3. chứng minh equicoercivity;
4. kết luận stability of minimizers.

# 10. Probability metrics trong generative models

| Metric | Ưu điểm | Nhược điểm |
|---|---|---|
| KL | phù hợp entropy/variational inference | không đối xứng, có thể vô hạn |
| TV | mạnh, diễn giải rõ | quá khắt khe trong high dimension |
| Wasserstein | hình học tốt, hợp OT | sample complexity xấu theo dimension |
| MMD | dễ estimate bằng kernel | phụ thuộc kernel |
| Fisher divergence | hợp score matching | cần score/log-density trơn |

Không có metric nào luôn tốt nhất. Chọn metric phụ thuộc câu hỏi nghiên cứu.

# 11. Liên hệ với diffusion models

Diffusion training học score \(s_\theta(x,t)\approx\nabla\log p_t(x)\). Nếu score error nhỏ theo chuẩn:

\[
\mathbb E_{t,x}\|s_\theta(x,t)-\nabla\log p_t(x)\|^2,
\]

thì ta hy vọng phân phối reverse SDE gần với \(\mu\). Câu hỏi lý thuyết là score error lan truyền thành distribution error như thế nào.

# 12. Liên hệ với flow matching

Flow matching học vector field \(v_\theta(x,t)\approx u_t(x)\). Nếu:

\[
\int_0^1\mathbb E_{p_t}\|v_\theta-u_t\|^2dt
\]

nhỏ, thì flow map sinh ra phân phối gần target. Câu hỏi thống kê là cần bao nhiêu mẫu để học \(u_t\) tốt.

# 13. Đề tài nghiên cứu từ probability/statistics

1. Statistical stability of entropy-regularized distributional optimization.
2. Finite-sample bounds for empirical KL-OT objectives.
3. Generalization of score matching under weak assumptions.
4. Wasserstein stability of flow matching under empirical measures.
5. Error decomposition for primal-dual diffusion alignment.
6. Concentration bounds for minibatch optimal transport objectives.

# 14. Bài tập tự kiểm tra

1. Chứng minh empirical average hội tụ với kỳ vọng thật cho hàm bị chặn.
2. Viết định nghĩa weak convergence và cho ví dụ.
3. So sánh TV và Wasserstein bằng ví dụ hai Dirac measures.
4. Code empirical measure của Gaussian mixture và tính MMD khi \(n\) tăng.
5. Viết một decomposition gồm statistical + optimization + approximation + sampling errors.
6. Giải thích vì sao score function cần Radon-Nikodym/log-density.

# 15. Kết luận

Probability cung cấp ngôn ngữ của phân phối. Statistics cung cấp câu trả lời cho câu hỏi học từ dữ liệu hữu hạn. Nếu thiếu hai mảng này, nghiên cứu generative model chỉ còn là tối ưu tham số. Nếu kết hợp probability/statistics với optimization và variational analysis, ta có một khung lý thuyết mạnh để phân tích diffusion và flow models.

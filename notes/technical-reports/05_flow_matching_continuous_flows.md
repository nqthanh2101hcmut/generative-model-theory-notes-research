---
title: "Technical Report 05 - Flow Matching and Continuous Flows"
subtitle: "ODE, Vector Fields, Continuity Equation and Optimal Transport Paths"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# 1. Từ diffusion sang flow

Diffusion models dùng stochastic process. Flow matching dùng deterministic ODE:

\[
\frac{dX_t}{dt}=v_t(X_t),
\qquad t\in[0,1].
\]

Nếu \(X_0\sim p_0\) và ODE đưa \(X_0\) thành \(X_1\), thì phân phối cuối kỳ vọng gần data distribution \(\mu\).

Flow matching học trực tiếp vector field \(v_t\), thay vì học score \(\nabla\log p_t\).

# 2. Flow map

Cho vector field \(v_t\), flow map \(\Phi_t\) thỏa:

\[
\frac{d}{dt}\Phi_t(x)=v_t(\Phi_t(x)),
\qquad
\Phi_0(x)=x.
\]

Nếu \(X_0\sim p_0\), thì:

\[
X_t=\Phi_t(X_0),
\qquad
p_t=(\Phi_t)_\#p_0.
\]

Đây là pushforward view.

# 3. Continuity equation

Mật độ \(p_t\) của flow thỏa:

\[
\partial_t p_t+\nabla\cdot(p_t v_t)=0.
\]

Phương trình này nói rằng khối lượng xác suất được bảo toàn khi di chuyển dưới vector field \(v_t\).

# 4. Continuous normalizing flows

Nếu \(v_\theta\) là neural network, continuous normalizing flow giải ODE:

\[
\frac{dX_t}{dt}=v_\theta(X_t,t).
\]

Log-density thay đổi theo:

\[
\frac{d}{dt}\log p_t(X_t)
=
-\nabla\cdot v_\theta(X_t,t).
\]

Training likelihood cần tính divergence, đôi khi tốn kém trong high dimension.

# 5. Flow matching objective

Flow matching tránh likelihood trực tiếp. Giả sử có path \(p_t\) và velocity field mục tiêu \(u_t\). Ta học:

\[
\min_\theta
\mathbb E_{t,x\sim p_t}
\|v_\theta(x,t)-u_t(x)\|^2.
\]

Nếu \(v_\theta\approx u_t\), ODE sinh ra phân phối gần path mục tiêu.

# 6. Conditional flow matching

Trong thực hành, chọn cặp \((x_0,x_1)\) với \(x_0\sim p_0\), \(x_1\sim\mu\), rồi định nghĩa:

\[
x_t=(1-t)x_0+tx_1.
\]

Conditional velocity:

\[
u_t=x_1-x_0.
\]

Loss:

\[
\mathcal L(\theta)
=
\mathbb E_{t,x_0,x_1}
\|v_\theta(x_t,t)-(x_1-x_0)\|^2.
\]

Đây là dạng đơn giản nhất.

# 7. Vai trò của coupling

Cặp \((x_0,x_1)\) được lấy theo coupling \(\pi\in\Pi(p_0,\mu)\). Nếu dùng independent coupling:

\[
\pi=p_0\otimes\mu,
\]

trajectory có thể dài và crossing nhiều. Nếu dùng OT coupling, trajectory thường ngắn và có cấu trúc tốt hơn.

# 8. OT-CFM

OT-CFM dùng optimal transport plan để ghép noise và data. Ý tưởng:

\[
\pi^\star
\in
\arg\min_{\pi\in\Pi(p_0,\mu)}
\int\|x_1-x_0\|^2d\pi(x_0,x_1).
\]

Sau đó dùng \((x_0,x_1)\sim\pi^\star\) trong conditional flow matching. Lợi ích:

- đường đi ngắn hơn;
- vector field đơn giản hơn;
- sampling bằng ODE có thể nhanh hơn;
- phù hợp với dynamic OT.

# 9. Stability of vector field approximation

Giả sử \(X_t\) giải ODE với vector field thật \(u_t\), và \(\widehat X_t\) giải ODE với vector field học được \(v_t\). Nếu:

\[
\|v_t(x)-u_t(x)\|\le\varepsilon
\]

và vector field Lipschitz với hằng số \(L\), thì Grönwall inequality cho bound dạng:

\[
\|\widehat X_t-X_t\|
\le
\frac{\varepsilon}{L}(e^{Lt}-1).
\]

Đây là công cụ cơ bản để biến vector field error thành trajectory error.

# 10. Distribution error

Nếu flow map sai, phân phối pushforward cũng sai. Mục tiêu là bound:

\[
W_p((\widehat\Phi_1)_\#p_0,(\Phi_1)_\#p_0)
\le
\text{function of }\|v-u\|.
\]

Câu hỏi nghiên cứu:

- cần chuẩn nào của \(v-u\)?
- expectation theo \(p_t\) có đủ không?
- Lipschitz assumptions có thực tế không?
- nếu data multimodal, trajectory crossing ảnh hưởng thế nào?

# 11. Numerical error

Flow sampling cần solve ODE. Với Euler:

\[
X_{k+1}=X_k+h v_\theta(X_k,t_k).
\]

Sai số phụ thuộc:

- step size \(h\);
- Lipschitz constant của vector field;
- smoothness theo time;
- stiffness của dynamics.

Một flow thẳng hơn thường cần ít bước ODE hơn.

# 12. Flow matching vs diffusion

| Tiêu chí | Diffusion | Flow matching |
|---|---|---|
| Dynamics | SDE hoặc probability flow ODE | ODE |
| Object learned | score | vector field |
| Sampling | thường nhiều bước | có thể nhanh hơn |
| Main math | SDE, Fisher divergence | ODE, continuity equation |
| OT connection | gián tiếp hoặc qua bridge | rất trực tiếp |
| Training | denoising/score matching | vector field regression |

# 13. Research questions

1. Path design nào làm vector field dễ học nhất?
2. Minibatch OT bias ảnh hưởng đến flow matching objective như thế nào?
3. Vector field approximation error lan truyền thành Wasserstein error ra sao?
4. Entropic OT regularization có cải thiện stability không?
5. Có thể thêm constraints vào flow matching bằng primal-dual không?
6. Flow matching có đạt statistical minimax rate trong setting nào?
7. ODE discretization error có thể tối ưu cùng training objective không?

# 14. Mini-project flow matching 2D

Các bước:

1. Tạo \(p_0=\mathcal N(0,I)\).
2. Tạo \(\mu\) là Gaussian mixture hoặc two moons.
3. Ghép random pairing và OT pairing.
4. Train MLP \(v_\theta(x,t)\).
5. Solve ODE từ \(t=0\) đến \(t=1\).
6. So sánh sample quality.
7. Vẽ vector field và trajectories.
8. Đo MMD/SWD.

# 15. Bài tập tự kiểm tra

1. Derive continuity equation từ conservation of mass.
2. Viết flow matching loss cho linear interpolation path.
3. Giải thích vai trò của coupling trong CFM.
4. Chứng minh bound trajectory bằng Grönwall trong trường hợp đơn giản.
5. Code Euler ODE solver cho vector field 2D.
6. So sánh flow matching với score matching bằng bảng.

# 16. Kết luận

Flow matching là hướng rất phù hợp cho người có nền tối ưu, ODE và optimal transport. Nó ít stochastic hơn diffusion nhưng lại mở ra nhiều câu hỏi toán học đẹp về path design, vector field stability, OT coupling, ODE discretization và Wasserstein convergence. Nếu muốn làm research theo hướng Toán ứng dụng, flow matching là một cửa vào rất tốt.

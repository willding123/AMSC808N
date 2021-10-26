% plot the objective function

fsz = 20;
figure;
plot(Gf,'Linewidth',2);
hold on 
plot(Lf,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('f','fontsize',fsz);
legend("Gauss-Newton", "Levenberg-Marquardt")
set(gca,'fontsize',fsz,'Yscale','log');
% plot the norm of the gradient
figure;
plot(Ggnorm,'Linewidth',2);
hold on 
plot(Lgnorm,'Linewidth',2);

xlabel('iter','fontsize',fsz);
ylabel('||g||','fontsize',fsz);
legend("Gauss-Newton", "Levenberg-Marquardt")

set(gca,'fontsize',fsz,'Yscale','log');
function y=gradient_comp_unsquared(alpha,beta,B,F,Z,p_out,p_in)
        H_now=B*F*((p_out/p_in)^Z-1);
        first_step_grad=2*alpha*H_now+beta;
        gradient_F=first_step_grad*B*((p_out/p_in)^Z-1);
        gradient_out=first_step_grad*B*F*Z/p_out*((p_out/p_in)^Z);
        gradient_in=-first_step_grad*B*F*Z/p_in*((p_out/p_in)^(Z/2));
        y=[gradient_F,gradient_out,gradient_in];
end
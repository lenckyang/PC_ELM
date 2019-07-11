


function [gx1min,MIN,gmin,T1]=PCOA(m,PP,TT,NumberofTrainingData,NumberofInputNeurons,p)
% function minimization
for count=1:1
%[x_best,t]=EM(20);
% -2.048 <=x1,x2<= 2.048
% min 100*(x1-x2).^2 +(1-x1).^2

a1=-1;
a2=0;
a3=0;

%b1=x_best(1)+0.5*x_best(1);
%b2=x_best(2)+0.5*x_best(2);
%b3=x_best(3)+0.5*x_best(3);
b1=1;
b2=pi;
b3=pi;
n1=NumberofInputNeurons+1;
p=2;
N1=ceil(m/p);
x1=a1+(b1-a1)*rand(n1,p);
x1(n1,:)=rand(1,p);

%x2=a2+(b2-a2)*rand(1,p);
%x3=a3+(b3-a3)*rand(1,p);



%x1=x_best(1)+0.03*rand(1,p);
%x2=x_best(2)+0.03*rand(1,p);
%x3=x_best(3)+0.03*rand(1,p);
xo1=x1;

%xo2=x2;
%xo3=x3;
    for j=1:n1
for i=1:p

    hx1(j,i)=sin(2/x1(j,i));
end
    end
    
x1min=x1;
%x2min=x2;
%x3min=x3;

for i=1:p

%lmin(i)=levy([x1(:,i)]);
%lmin(i)=rosen([x1(:,i)]);
%lmin(i)=ackley([x1(:,i)]);
%lmin(i)=zakh([x1(:,i)]);
I_InputWeight=x1(1:n1-1,i);
I_BiasofHiddenNeurons=x1(n1,i);
ind=ones(1,NumberofTrainingData);
BiasMatrix=I_BiasofHiddenNeurons(:,ind);  
I_tempH=I_InputWeight'*PP;

II_tempH=I_tempH+BiasMatrix;
H = 1 ./ (1 + exp(-II_tempH));
T1=TT;
 
OutputWeight=pinv(H') * T1';      
T1=T1-(H' * OutputWeight)'; 
lmin(i)=sqrt(mse(T1));
%lmin(i)=easom_2([x1(i),x2(i)]);
%lmin(i)=gold_2([x1(i),x2(i)]);
%lmin(i)=griewank([x1(:,i)]);
%lmin(i)=hart3([x1(i),x2(i),x3(i)]);
%lmin(i)=mich([x1(:,i)]);
%lmin(i)=rast([x1(:,i)]);
%lmin(i)=schw([x1(:,i)]);
%lmin(i)=sphere([x1(i),x2(i),x3(i)]);
%lmin(i)=trid([x1(:,i)]);
%lmin(i)=zakh([x1(:,i)]);
end

gmin=lmin(1);
fgmin=lmin(1);
gx1min=x1min(:,1);
for i=1:p
    if lmin(i)<gmin
        gmin=lmin(i);
        for j=1:n1
        gx1min=x1min(:,i);
        end
     end
end

        
MIN=[];

for k=1:N1
    if k==1 
        ac1=a1*ones(n1,p);
       % ac2=a2*ones(1,p);
       % ac3=a3*ones(1,p);
        bc1=b1*ones(n1,p);
       % bc2=b2*ones(1,p);
       % bc3=b3*ones(1,p);
    end
   for j=1:n1
    for i=1:p
       if k>ceil(m/p);
                       gx11min(j,i)=gx1min(j,1)+0.001*(0.5-rand);
           % x1(j,i)=gx1min(j,i)+q*gx1min(j,i)*(0.5-abs(hx1(j,i)));
             x1(j,i)=gx11min(j,i)+q*(abs(hx1(j,i))-0.5);
       else
        x1(j,i)=ac1(j,i)+(bc1(j,i)-ac1(j,i))*abs(hx1(j,i));
       end
        if x1(j,i)>1
            x1(j,i)=1;
        end
        if x1(j,i)<-1
            x1(j,i)=-1;
        end
        %x2(i)=ac2(i)+(bc2(i)-ac2(i))*abs(hx2(i));
       % x3(i)=ac3(i)+(bc3(i)-ac3(i))*abs(hx3(i));
    end
   end
    %fmin=100*(x1-x2).^2+(1*ones(1,p)-x1).^2+x3.^2;
    for i=1:p
%fmin(i)=levy([x1(:,i)]);
%fmin(i)=rosen([x1(:,i)]);
%fmin(i)=ackley([x1(:,i)]);
%fmin(i)=zakh([x1(:,i)]);
I_InputWeight=x1(1:n1-1,i);
I_BiasofHiddenNeurons=x1(n1,i);
ind=ones(1,NumberofTrainingData);
BiasMatrix=I_BiasofHiddenNeurons(:,ind);  
I_tempH=I_InputWeight'*PP;

II_tempH=I_tempH+BiasMatrix;
H = 1 ./ (1 + exp(-II_tempH));
T1=TT;

OutputWeight=pinv(H') * T1';      
T1=T1-(H' * OutputWeight)' ; 
fmin(i)=sqrt(mse(T1));
%fmin(i)=easom_2([x1(:,i)]);
%fmin(i)=gold_2([x1(:,i)]);
%fmin(i)=griewank([x1(:,i)]);
%fmin(i)=hart3([x1(:,i)]);

%fmin(i)=mich([x1(:,i)]);
%fmin(i)=rast([x1(:,i)]);
%fmin(i)=schw([x1(i),x2(i),x3(i)]);
%fmin(i)=sphere([x1(:,i)]);
%fmin(i)=trid([x1(:,i)]);
%fmin(i)=zakh([x1(:,i)]);
    end
    
    for i=1:p
            if fmin(i)<fgmin
                fgmin=fmin(i);
                for j=1:n1
                fx1min=x1(:,i);
               % fx2min=x2(i);
              %  fx3min=x3(i);
                end
            end
        end
    
    
     for i=1:p
            if fmin(i)<lmin(i)
                lmin(i)=fmin(i);
                for j=1:n1
                x1min=x1(:,i);
                end
               % x2min(i)=x2(i);
              %  x3min(i)=x3(i);
            end
        end
  
            if fgmin<gmin
                gmin=fgmin;
                for j=1:n1
                gx1min=fx1min;
                end
             %   gx3min=fx3min;
            end
 
   if k>ceil(m/p);
       q=1-(10*(k*10/10-1)/k*10).^0.07;
   else q=1;
   end
   for j=1:n1
    ac1(j,:)=x1min(j,:)-q*(b1-a1)*ones(1,p);
  bc1(j,:)=x1min(j,:)+q*(b1-a1)*ones(1,p);
   end
   
  %  ac2=x2min-q*(b2-a2)*ones(1,p);
   % bc2=x2min+q*(b2-a2)*ones(1,p);
  %  ac3=x3min-q*(b3-a3)*ones(1,p);
 %   bc3=x3min+q*(b3-a3)*ones(1,p);
 for j=1:n1
    for i=1:p
        if ac1(j,i)<a1   ac1(j,i)=a1; end
        if bc1(j,i)>b1   bc1(j,i)=b1; end
    end
    end
for j=1:n1
for i=1:p

    hx1(j,i)=sin(2/hx1(j,i));
end
end

   MIN=[MIN gmin];
      
      
   end
end
%MIN=[t,MIN];
%hold on
%plot(MIN,'r')
%{
hold on
plot(MIN,'b')
hold on
for i=1:N1
    if MIN(i)<0.0001
  goal_call(count)=i;
  min_ff(count)=MIN(i);
      plot(i,MIN(i),'o');
        break;
    end
    
end
%}
end


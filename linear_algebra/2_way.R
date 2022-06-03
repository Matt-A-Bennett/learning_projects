n=5; r=2; c=3;
Example.Data<-data.frame(ID=seq(1:(n*r*c)),
          Level=c(rep("Low",(n*c)),rep("High",(n*c))),
          Type=rep(c(rep("Watching",n),rep("Reading",n),rep("Thinking",n)),2),
          Percieved.Ability = c(1,3,3,4,2,3,2,3,1,4,3,3,2,3,1,
                                6,4,7,5,5,5,3,2,3,2,3,2,1,4,4))

# Also, I want to re-set the order using the `factor` command and set the order I want. 
Example.Data$Level<-factor(Example.Data$Level, 
                              levels = c("Low","High"),
                              labels = c("Low","High"))

Example.Data$Type<-factor(Example.Data$Type, 
                              levels = c("Watching","Reading","Thinking"),
                              labels = c("Watching","Reading","Thinking"))


library(dplyr)
Means.Table<-Example.Data %>%
  group_by(Level,Type) %>%
  summarise(N=n(),
            Means=mean(Percieved.Ability),
            SS=sum((Percieved.Ability-Means)^2),
            SD=sd(Percieved.Ability),
            SEM=SD/N^.5)
knitr::kable(Means.Table, digits = 2) 
# Note remember knitr::kable makes it pretty, but you can just call `Means.Table`

library(ggplot2)

Plot.1<-ggplot(Means.Table, aes(x = Type, y = Means, group=Level))+
  geom_col(aes(group=Level, fill=Level), position=position_dodge(width=0.9))+
  scale_y_continuous(expand = c(0, 0))+ # Forces plot to start at zero
  geom_errorbar(aes(ymax = Means + SEM, ymin= Means - SEM), 
                position=position_dodge(width=0.9), width=0.25)+
  scale_fill_manual(values=c("#f44141","#4286f4"))+
  xlab('')+
  ylab('Percieved Ability')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank(),
        axis.line=element_line(),
        legend.title=element_blank())
Plot.1

Example.Data

ANOVA.JA.Table<-anova(lm(Percieved.Ability~Level*Type,
                  data=Example.Data))
ANOVA.JA.Table

cancer <- array(data = c(35, 42, 47, 26, 59, 77, 112, 76), 
                dim = c(2,2,2), 
                dimnames = list("center" = c("Boston","Glamorgan"),
                                "grade" = c("malignant","benign"),
                                "survival" = c("died","survived")))

cancer.df <- as.data.frame(as.table(cancer))
cancer.df[,1] <- relevel(cancer.df[,1], ref = "Boston")
cancer.df[,2] <- relevel(cancer.df[,2], ref = "malignant")
cancer.df[,3] <- relevel(cancer.df[,3], ref = "died")

mod0 <- glm(Freq ~ center + grade + survival + center * grade + 
              center * survival + grade * survival + center * grade * survival, 
            data = cancer.df, family = poisson)
summary(mod0)
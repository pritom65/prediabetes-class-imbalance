library(tidyverse)
library(tidymodels)
library(themis)
library(future)

df <- read_csv("data/Dataset of Diabetes.csv") |> 
  select(-c(ID, No_Pation)) |> 
  mutate(Gender = tolower(Gender),
         CLASS = str_trim(CLASS)) |> 
  distinct()



df_split <- initial_split(df,prop = .75, strata = CLASS)
df_vfold <- vfold_cv(training(df_split),strata = CLASS)

eval_mat <- metric_set(accuracy,bal_accuracy,f_meas,precision,recall,spec,sens)


plan(multisession, workers = parallel::detectCores() - 2)
mn_model <- multinom_reg(mode = "classification") |> set_engine("nnet")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) 

temp_wf <- workflow() |> 
  add_model(spec = mn_model) |> 
  add_recipe(df_recipe_up) 

mn_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = mn_model,file = "./model/model_mn_baseline.rds")



mn_model <- multinom_reg(mode = "classification") |> set_engine("nnet")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |>
  step_upsample(CLASS,over_ratio = tune() )

temp_wf <- workflow() |> 
  add_model(spec = mn_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1))), size = 10)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE),metrics = metric_set(roc_auc))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

mn_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = mn_model,file = "./model/model_mn_over.rds")



mn_model <- multinom_reg(mode = "classification") |> set_engine("nnet")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_downsample(CLASS,under_ratio = tune() )

temp_wf <- workflow() |> 
  add_model(spec = mn_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(under_ratio(range = c(1, 4))), size = 10)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

mn_model <- temp_wf |> 
  last_fit(split = df_split)
write_rds(x = mn_model,file = "./model/model_mn_under.rds")



mn_model <- multinom_reg(mode = "classification") |> set_engine("nnet")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_dummy(Gender) |> 
  step_range(all_numeric_predictors()) |> 
  step_smote(CLASS,over_ratio = tune(),neighbors = tune() )

temp_wf <- workflow() |> 
  add_model(spec = mn_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1)),
                                   neighbors(c(2,10))),
                    size = 15)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

mn_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = mn_model,file = "./model/model_mn_smote.rds")



mn_model <- multinom_reg(mode = "classification") |> set_engine("nnet")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_dummy(Gender) |> 
  step_range(all_numeric_predictors()) |> 
  step_adasyn(CLASS,over_ratio = tune(),neighbors = tune() )

temp_wf <- workflow() |> 
  add_model(spec = mn_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1)),
                                   neighbors(c(2,10))),
                    size = 15)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

mn_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = mn_model,file = "./model/model_mn_adasyn.rds")



library(discrim)

plan(multisession, workers = parallel::detectCores() - 2)
qda_model <- discrim_quad(mode = "classification") |> set_engine("MASS")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) 

temp_wf <- workflow() |> 
  add_model(spec = qda_model) |> 
  add_recipe(df_recipe_up) 

qda_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = qda_model,file = "./model/model_qda_baseline.rds")



qda_model <- discrim_quad(mode = "classification") |> set_engine("MASS")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |>
  step_upsample(CLASS,over_ratio = tune() )

temp_wf <- workflow() |> 
  add_model(spec = qda_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1))), size = 10)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE),metrics = metric_set(roc_auc))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

qda_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = qda_model,file = "./model/model_qda_over.rds")


qda_model <- discrim_quad(mode = "classification") |> set_engine("MASS")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_downsample(CLASS,under_ratio = tune() )

temp_wf <- workflow() |> 
  add_model(spec = qda_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(under_ratio(range = c(1, 4))), size = 10)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

qda_model <- temp_wf |> 
  last_fit(split = df_split)
write_rds(x = qda_model,file = "./model/model_qda_under.rds")



qda_model <- discrim_quad(mode = "classification") |> set_engine("MASS")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_dummy(Gender) |> 
  step_range(all_numeric_predictors()) |> 
  step_smote(CLASS,over_ratio = tune(),neighbors = tune() )

temp_wf <- workflow() |> 
  add_model(spec = qda_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1)),
                                   neighbors(c(2,10))),
                    size = 15)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

qda_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = qda_model,file = "./model/model_qda_smote.rds")


qda_model <- discrim_quad(mode = "classification") |> set_engine("MASS")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_dummy(Gender) |> 
  step_range(all_numeric_predictors()) |> 
  step_adasyn(CLASS,over_ratio = tune(),neighbors = tune() )

temp_wf <- workflow() |> 
  add_model(spec = qda_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1)),
                                   neighbors(c(2,10))),
                    size = 15)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

qda_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = qda_model,file = "./model/model_qda_adasyn.rds")
detach("package:discrim", unload = TRUE, character.only = TRUE)



dt_model <- decision_tree(mode = "classification", cost_complexity = tune(), tree_depth = tune(), min_n = tune()) |> set_engine("rpart")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) 

temp_wf <- workflow() |> 
  add_model(spec = dt_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(cost_complexity(), tree_depth(), min_n()), size = 10)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE),metrics = metric_set(roc_auc))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

dt_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = dt_model,file = "./model/model_dt_baseline.rds")


dt_model <- decision_tree(mode = "classification", cost_complexity = tune(), tree_depth = tune(), min_n = tune()) |> set_engine("rpart")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |>
  step_upsample(CLASS,over_ratio = tune())

temp_wf <- workflow() |> 
  add_model(spec = dt_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1)),
                                   cost_complexity(), tree_depth(), min_n()),
                    size = 10)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE),metrics = metric_set(roc_auc))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

dt_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = dt_model,file = "./model/model_dt_over.rds")


dt_model <- decision_tree(mode = "classification", cost_complexity = tune(), tree_depth = tune(), min_n = tune()) |> set_engine("rpart")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_downsample(CLASS,under_ratio = tune())

temp_wf <- workflow() |> 
  add_model(spec = dt_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(under_ratio(range = c(1, 4)), cost_complexity(), tree_depth(), min_n()), size = 10)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

dt_model <- temp_wf |> 
  last_fit(split = df_split)
write_rds(x = dt_model,file = "./model/model_dt_under.rds")


dt_model <- decision_tree(mode = "classification", cost_complexity = tune(), tree_depth = tune(), min_n = tune()) |> set_engine("rpart")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_dummy(Gender) |> 
  step_range(all_numeric_predictors()) |> 
  step_smote(CLASS,over_ratio = tune(),neighbors = tune())

temp_wf <- workflow() |> 
  add_model(spec = dt_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1)),
                                   neighbors(c(2,10)),
                                   cost_complexity(), tree_depth(), min_n()),
                    size = 15)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

dt_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = dt_model,file = "./model/model_dt_smote.rds")


dt_model <- decision_tree(mode = "classification", cost_complexity = tune(), tree_depth = tune(), min_n = tune()) |> set_engine("rpart")


df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_dummy(Gender) |> 
  step_range(all_numeric_predictors()) |> 
  step_adasyn(CLASS,over_ratio = tune(),neighbors = tune() )

temp_wf <- workflow() |> 
  add_model(spec = dt_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1)),
                                   neighbors(c(2,10)),
                                   cost_complexity(), tree_depth(), min_n()),
                    size = 15)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

dt_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = dt_model,file = "./model/model_dt_adasyn.rds")


rf_model <- rand_forest(mode = "classification",min_n = tune(),mtry = tune(),trees = 500) |> set_engine("ranger", importance = "impurity")

df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) 


temp_wf <- workflow() |> 
  add_model(spec = rf_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(mtry(c(2,8)), min_n()),size = 10)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE),metrics = metric_set(roc_auc))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

rf_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = rf_model,file = "./model/model_rf_baseline.rds")


rf_model <- rand_forest(mode = "classification",min_n = tune(),mtry = tune(),trees = 1500) |> set_engine("ranger", importance = "impurity")

df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |>
  step_upsample(CLASS,over_ratio = tune() )


temp_wf <- workflow() |> 
  add_model(spec = rf_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1)),
                                   mtry(c(2,8)),
                                   min_n()),
                    size = 10
)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE),metrics = metric_set(roc_auc))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

rf_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = rf_model,file = "./model/model_rf_over.rds")


rf_model <- rand_forest(mode = "classification",min_n = tune(),mtry = tune(),trees = 1500)

df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_downsample(CLASS,under_ratio = tune() )


temp_wf <- workflow() |> 
  add_model(spec = rf_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(under_ratio(range = c(1, 4)), mtry(c(2, 8)), min_n(c(10, 30))), size = 10)

grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

rf_model <- temp_wf |> 
  last_fit(split = df_split)
write_rds(x = rf_model,file = "./model/model_rf_under.rds")


rf_model <- rand_forest(mode = "classification",min_n = tune(),mtry = tune(),trees = 1500)

df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_dummy(Gender) |> 
  step_range(all_numeric_predictors()) |> 
  step_smote(CLASS,over_ratio = tune(),neighbors = tune() )


temp_wf <- workflow() |> 
  add_model(spec = rf_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1)),
                                   neighbors(c(2,10)),
                                   mtry(c(2,8)),
                                   min_n()),
                    size = 15
)


grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

rf_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = rf_model,file = "./model/model_rf_smote.rds")


rf_model <- rand_forest(mode = "classification",min_n = tune(),mtry = tune(),trees = 1500)

df_recipe_up <- recipe(x = training(df_split),formula = CLASS ~ .) |> 
  step_string2factor(CLASS,Gender) |> 
  step_dummy(Gender) |> 
  step_range(all_numeric_predictors()) |> 
  step_adasyn(CLASS,over_ratio = tune(),neighbors = tune() )


temp_wf <- workflow() |> 
  add_model(spec = rf_model) |> 
  add_recipe(df_recipe_up) 


grid <- grid_random(x = parameters(over_ratio(range = c(.5,1)),
                                   neighbors(c(2,10)),
                                   mtry(c(2,8)),
                                   min_n()),
                    size = 15
)


grid_result <- tune_grid(object = temp_wf,resamples = df_vfold,grid = grid,control = control_grid(verbose = TRUE,save_pred = TRUE))

temp_wf <- temp_wf |> 
  finalize_workflow(parameters = select_best(grid_result))

rf_model <- temp_wf |> 
  last_fit(split = df_split)

write_rds(x = rf_model,file = "./model/model_rf_adasyn.rds")



df_model <- tibble(model_config = list.files("./model/")) |> 
  mutate(model = str_extract(string = model_config,pattern = "_.*_"),
         model = str_remove_all(model,"_"),
         config = str_extract(model_config,pattern = "_[a-z]+\\."),
         config = str_remove_all(config,"\\.|_")) |> 
  mutate(model_performance = map(.x = model_config,.f = \(model_name) {
    model <- read_rds(here::here(".","model",model_name))
    model |> 
      collect_predictions() |> 
      roc_auc(truth = CLASS,
              c(.pred_N, .pred_P, .pred_Y), 
              estimator = "macro_weighted") |> 
      bind_rows(
        collect_predictions(model) |> 
          eval_mat(truth = CLASS,estimate = .pred_class)
      )}
  )) |> 
  unnest()


df_model |> 
  select(model,config,.metric,.estimate) |>
  mutate(config = factor(config,c("baseline","under","over","smote","adasyn"))) |> 
  ggplot(aes(config,.estimate,col = model)) +
  geom_point() +
  geom_line(aes(group = model), alpha = .5) +
  facet_wrap(~.metric) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30,vjust = .9),
        legend.position = c(.8,.05)) 

ggsave(filename = "plotForModel.pdf")
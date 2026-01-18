#include <stdio.h>
#include <stdlib.h>
#include<math.h>

typedef struct {
    int epoch;
    double lr;
    double *theta;
} LinearRegression;

void init(LinearRegression *model, int epoch, double lr, int n_feature) {
    model->epoch = epoch;
    model->lr = lr;

    model->theta = (double*) malloc(sizeof(double) * (n_feature + 1));
    if (!model->theta) {
        printf("Theta allocation failed\n");
        exit(1);
    }

    for (int i = 0; i <= n_feature; i++) {
        model->theta[i] = 0.0;
    }
}

double dot(const double *theta, const double *x_row, int n_feature) {
    double sum = theta[0];
    for (int j = 0; j < n_feature; j++) {
        sum += theta[j + 1] * x_row[j];
    }
    return sum;
}

void predict(LinearRegression *model, double **X, double *y_pred, int m, int n) {
    for (int i = 0; i < m; i++) {
        y_pred[i] = dot(model->theta, X[i], n);
    }
}

double loss(const double *y_pred, const double *y, int m) {
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        double err = y_pred[i] - y[i];
        sum += err * err;
    }
    return sum / m;
}

void train(LinearRegression *model, double **X, double *y, int m, int n, FILE *loss_fp) {

    double *y_pred = (double*) malloc(sizeof(double) * m);

    for (int epoch = 0; epoch < model->epoch; epoch++) {

        predict(model, X, y_pred, m, n);

        double l = loss(y_pred, y, m);

        fprintf(loss_fp, "%d,%f\n", epoch, l);

        if (epoch % 100 == 0) {
            printf("Epoch %d , Loss = %f\n", epoch, l);
        }

        double *grad = (double*) calloc(n + 1, sizeof(double));

        for (int i = 0; i < m; i++) {
            double error = y_pred[i] - y[i];
            grad[0] += error;

            for (int j = 0; j < n; j++) {
                grad[j + 1] += error * X[i][j];
            }
        }

        for (int j = 0; j <= n; j++) {
            model->theta[j] -= (model->lr * 2.0 / m) * grad[j];
        }

        free(grad);
    }

    free(y_pred);
}

void free_model(LinearRegression *model) {
    free(model->theta);
}
void standardize(double **X, int m, int n) {
    for (int j = 0; j < n; j++) {
        double mean = 0.0;
        double std = 0.0;

        for (int i = 0; i < m; i++)
            mean += X[i][j];
        mean /= m;

        for (int i = 0; i < m; i++)
            std += (X[i][j] - mean) * (X[i][j] - mean);
        std = sqrt(std / m);

        if (std == 0.0) std = 1.0;

        for (int i = 0; i < m; i++)
            X[i][j] = (X[i][j] - mean) / std;
    }
}


int main() {

    int m = 5;
    int n = 2;

    double **X = (double**) malloc(m * sizeof(double*));
    for (int i = 0; i < m; i++)
        X[i] = (double*) malloc(n * sizeof(double));

    double *y = (double*) malloc(m * sizeof(double));

    X[0][0] = 1; X[0][1] = 2; y[0] = 13;
    X[1][0] = 2; X[1][1] = 1; y[1] = 12;
    X[2][0] = 3; X[2][1] = 4; y[2] = 23;
    X[3][0] = 4; X[3][1] = 3; y[3] = 22;
    X[4][0] = 5; X[4][1] = 5; y[4] = 30;

    LinearRegression model;
    init(&model, 3000, 0.01, n);

    FILE *loss_fp = fopen("loss.csv", "w");
    if (!loss_fp) {
        printf("Failed to open loss file\n");
        return 1;
    }
    fprintf(loss_fp, "epoch,loss\n");

    train(&model, X, y, m, n, loss_fp);

    fclose(loss_fp);

    printf("\nLearned parameters:\n");
    printf("Bias (theta[0]) = %f\n", model.theta[0]);
    for (int i = 1; i <= n; i++) {
        printf("Theta[%d] = %f\n", i, model.theta[i]);
    }

    double *y_pred = (double*) malloc(sizeof(double) * m);
    predict(&model, X, y_pred, m, n);

    printf("\nPredictions:\n");
    for (int i = 0; i < m; i++) {
        printf("y_pred[%d] = %.2f | y_true[%d] = %.2f\n",
               i, y_pred[i], i, y[i]);
    }

    FILE *pred_fp = fopen("predictions.csv", "w");
    fprintf(pred_fp, "y_true,y_pred\n");
    for (int i = 0; i < m; i++) {
        fprintf(pred_fp, "%f,%f\n", y[i], y_pred[i]);
    }
    fclose(pred_fp);

    free(y_pred);
    free_model(&model);

    for (int i = 0; i < m; i++)
        free(X[i]);
    free(X);
    free(y);

    return 0;
}
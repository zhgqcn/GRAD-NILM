import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def _normalized_disaggregation_error(y, y_hat):
    """
        Function that computes the normalized disaggregation error (NDE)

        Arguments:
            y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
            y (torch.Tensor) : Shape (B x T x M) ground truth targets
        Returns:
            NDE (float): normalized disaggregation error
    """
    return torch.sum((y_hat - y) ** 2) / torch.sum(y ** 2)


def _estimated_accuracy(y, y_hat):
    """
       Function that computes the estimated accuracy (EAC)

       Arguments:
           y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
           y (torch.Tensor) : Shape (B x T x M) ground truth targets
       Returns:
           EAC (float): estimated accuracy
    """
    eac = 1. - (y_hat - y).abs().sum(dim=1).mean() / (2. * y.abs().sum(dim=1)).mean()
    return np.where(eac<0, 0, eac)


def _compute_f1(s, s_hat):
    """
       Function that computes true positives (tp), false positives (fp) and
       false negatives (fn).

       Arguments:
           s_hat (torch.Tensor) : Shape (B x T x M) model state predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth states
       Returns:
           tp (int), fp (int), fn (int) : Tuple[int] containing tp, fp and fn
    """
    tp = torch.sum(s * s_hat).float()
    fp = torch.sum(torch.logical_not(s) * s_hat).float()
    fn = torch.sum(s * torch.logical_not(s_hat)).float()

    return tp, fp, fn


def example_f1_score(s, s_hat):
    """
       Function that computes the example-based F1-score (eb-F1)

       Arguments:
           s_hat (torch.Tensor) : Shape (B x T x M) model state predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth states
       Returns:
           eb-F1 (float): example-based F1-score
    """

    tp, fp, fn = _compute_f1(s, s_hat)
    numerator = 2 * tp
    denominator = torch.sum(s).float() + torch.sum(s_hat).float()
    return numerator / (denominator + 1e-12)


def _mae(y, y_hat):
    """
       Function that computes the mean absolute error (MAE)

       Arguments:
           y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
           y (torch.Tensor) : Shape (B x T x M) ground truth targets
       Returns:
           MAE (float): mean absolute error
    """
    return torch.mean(torch.abs(y - y_hat))


def per_appliance_metrics(y_hat, y, s_hat, s):
    """
       Function that computes the F1-Score, NDE, EAC and MAE per appliance

       Arguments:
           y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
           y (torch.Tensor) : Shape (B x T x M) ground truth targets
           s_hat (torch.Tensor) : Shape (B x T x M) model on/off predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth on/off targets
       Returns:
           metrics (torch.Tensor): Shape (M x 4) Matrix containing each metric (F1, NDE, EAC, MAE) for each appliance
    """
    # assumes that y_hat is the median prediction and that y_hat.shape == y.shape
    assert y_hat.shape == y.shape
    assert s_hat.shape == s.shape
    tensors = []
    s_hat_clone = s_hat[:]
    s_hat_clone[s_hat < 0.5] = 0
    s_hat_clone[s_hat >= 0.5] = 1
    for appliance in range(y_hat.shape[-1]):
        # try:
        #     f1 = f1_score(s[..., appliance], s_hat_clone[..., appliance], average='samples', zero_division=0)
        # except ValueError:
        #     print(s_hat_clone[torch.logical_and(s_hat_clone > 0, s_hat_clone < 1)])
        f1 = example_f1_score(s[..., appliance], s_hat[..., appliance])
        nde = _normalized_disaggregation_error(y[..., appliance], y_hat[..., appliance])
        eac = _estimated_accuracy(y[..., appliance], y_hat[..., appliance])
        mae = _mae(y[..., appliance], y_hat[..., appliance])
        tensors.append(torch.tensor([f1, nde, eac, mae]))
    return torch.stack(tensors)


def mean_metrics(y_hat, y, s_hat, s):
    """
       Function that computes the F1-Score, NDE, EAC and MAE per appliance

       Arguments:
           y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
           y (torch.Tensor) : Shape (B x T x M) ground truth targets
           s_hat (torch.Tensor) : Shape (B x T x M) model on/off predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth on/off targets
       Returns:
           metrics (torch.Tensor): Shape (4, ) Vector containing the mean of each metric (F1, NDE, EAC, MAE)
    """
    return torch.mean(per_appliance_metrics(y_hat, y, s_hat, s), dim=0)


def per_appliance_metrics_pandas(y_hat, y, s_hat, s, appliances=None, metrics=None):
    """
       Function that computes the F1-Score, NDE, EAC and MAE per appliance and returns
       the results as dataframe

       Arguments:
           y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
           y (torch.Tensor) : Shape (B x T x M) ground truth targets
           s_hat (torch.Tensor) : Shape (B x T x M) model on/off predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth on/off targets
           appliances (List[String]) : List of appliance names
           metrics (List[String]) : List of metrics
       Returns:
           frame (pandas.DataFrame): Shape (M x 4) Matrix containing each metric for each appliance
    """
    if metrics is None:
        metrics = ['F1', 'NDE', 'EAC', 'MAE']
    if appliances is None:
        appliances = ['washing_machine', 'dishwasher', 'kettle', 'fridge', 'microwave']

    per_appl_tensor = per_appliance_metrics(y_hat, y, s_hat, s)
    frame = pd.DataFrame(columns=['Appliance'] + metrics)
    frame['Appliance'] = appliances
    frame = frame.set_index('Appliance')
    for i, app in enumerate(appliances):
        frame.loc[app] = per_appl_tensor[i]
    return frame


def mean_metrics_pandas(y_hat, y, s_hat, s, appliances=None, metrics=None):
    """
          Function that computes the F1-Score, NDE, EAC and MAE per appliance and returns
          the results as dataframe

          Arguments:
              y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
              y (torch.Tensor) : Shape (B x T x M) ground truth targets
              s_hat (torch.Tensor) : Shape (B x T x M) model on/off predictions
              s (torch.Tensor) : Shape (B x T x M) ground truth on/off targets
              appliances (List[String]) : List of appliance names
              metrics (List[String]) : List of metrics
          Returns:
              frame (pandas.DataFrame): Shape (M x 4) Matrix containing the mean of each metric for each appliance
   """
    if metrics is None:
        metrics = ['F1', 'NDE', 'EAC', 'MAE']
    if appliances is None:
        appliances = ['washing_machine', 'dishwasher', 'kettle', 'fridge', 'microwave']

    frame = per_appliance_metrics_pandas(y_hat, y, s_hat, s, appliances, metrics)
    return frame.mean()


def acc_Power(x_pred, x_test, c_test=0, app_ratio=0, disaggregation=False):
    if disaggregation:
        Pest = np.sum(x_pred, axis=1).reshape([-1,1])

        Preal = np.sum(x_test, axis=1).reshape([-1,1])

        acc_P = ((np.abs(Pest - Preal)/(2*Preal))*-1)+1

        acc_P = np.nan_to_num(acc_P)
        
        acc_P_tot = np.mean(acc_P[acc_P>0])
        acc_P_app = acc_P.reshape(-1)
    else:
        M_ratio = (np.tile(app_ratio, [c_test.shape[0],1])*c_test)

        Pest = np.sum(x_pred, axis=1).reshape([-1,1])

        Pest = (np.tile(Pest, [1,c_test.shape[1]])*M_ratio)

        Preal = np.sum(x_test, axis=1).reshape([-1,1])

        Preal = (np.tile(Preal, [1,c_test.shape[1]])*M_ratio)

        acc_P = ((np.abs(Pest - Preal)/(2*Preal))*-1)+1

        acc_P = np.nan_to_num(acc_P)

        acc_P_tot = acc_P.sum(axis=1).mean()
        acc_P_app = acc_P.sum(axis=0)/c_test.sum(axis=0)

    print(acc_P_tot)
    
    return acc_P_tot, acc_P_app, acc_P

def MAE_metric(x_pred, x_test, c_test=0, app_ratio=0, disaggregation=False, only_power_on=False):
    
    if disaggregation:
        if only_power_on:
            MAE = np.zeros(x_pred.shape[0])
            for i in range(x_pred.shape[0]):
                ind = (x_pred[i,:])>0
                MAE[i] = np.mean(np.abs((x_test[i,ind]-x_pred[i,ind])))
            MAE = np.nan_to_num(MAE)
        else:
            MAE = np.mean(np.abs((x_test-x_pred)), axis=1).reshape([-1,1])
        MAE_app = MAE
        MAE_tot = np.mean(MAE[MAE>0])
    else:
        MAE = np.mean(np.abs((x_test-x_pred)), axis=1).reshape([-1,1])
        
        M_ratio = (np.tile(app_ratio, [c_test.shape[0],1])*c_test)
        MAE = np.tile(MAE, [1,c_test.shape[1]])*M_ratio

        MAE_tot = MAE.sum(axis=1).mean()
        MAE_app = MAE.sum(axis=0)/c_test.sum(axis=0)

    print(MAE_tot)
    
    return MAE_tot, MAE_app, MAE

def SAE_metric(x_pred, x_test):
    
    SAE = np.zeros(x_pred.shape[0])
    
    for i in range(x_pred.shape[0]):
        SAE[i] = np.abs(x_pred[i,:].sum() - x_test[i,:].sum())/x_test[i,:].sum()
        
    print(SAE)
    
    return SAE

def EpD_metric(x_pred, x_test, sampling=6):
    
    sPerDay = (60//sampling)*60*24
    
    EpD = np.zeros(x_pred.shape[0])
    
    for i in range(x_pred.shape[0]):
        N_days = x_pred[i,:].shape[0]//sPerDay
        EpD[i] = np.mean(np.abs(np.sum(x_pred[i,0:N_days*sPerDay].reshape(N_days,-1), axis=-1)-np.sum(x_test[i,0:N_days*sPerDay].reshape(N_days,-1), axis=-1)))*sampling/3600
        
    print(EpD)
    
    return EpD

def F1_metric(x_pred, x_test, thr):
    from sklearn.metrics import f1_score as f1_score
    
    x_pred_b = np.copy(x_pred)
    x_pred_b[x_pred_b<thr] = 0
    x_pred_b[x_pred_b>=thr] = 1
    
    x_test_b = np.copy(x_test)
    x_test_b[x_test_b<thr] = 0
    x_test_b[x_test_b>=thr] = 1
    
    F1 = np.zeros(x_pred.shape[0])
    
    for i in range(x_pred.shape[0]):
        F1[i] = f1_score(x_test_b[i,:], x_pred_b[i,:])

    for s in F1:
        print(s)
        
    return F1

def RE_metric(x_pred, x_test, thr):
    from sklearn.metrics import recall_score
    
    x_pred_b = np.copy(x_pred)
    x_pred_b[x_pred_b<thr] = 0
    x_pred_b[x_pred_b>=thr] = 1
    
    x_test_b = np.copy(x_test)
    x_test_b[x_test_b<thr] = 0
    x_test_b[x_test_b>=thr] = 1
    
    RE = np.zeros(x_pred.shape[0])
    
    for i in range(x_pred.shape[0]):
        RE[i] = recall_score(x_test_b[i,:], x_pred_b[i,:])
        
    for s in RE:
        print(s)
        
    return RE

def PR_metric(x_pred, x_test, thr):
    from sklearn.metrics import precision_score
    
    x_pred_b = np.copy(x_pred)
    x_pred_b[x_pred_b<thr] = 0
    x_pred_b[x_pred_b>=thr] = 1
    
    x_test_b = np.copy(x_test)
    x_test_b[x_test_b<thr] = 0
    x_test_b[x_test_b>=thr] = 1
    
    PR = np.zeros(x_pred.shape[0])
    
    for i in range(x_pred.shape[0]):
        PR[i] = precision_score(x_test_b[i,:], x_pred_b[i,:])
        
    for s in PR:
        print(s)
        
    return PR


def RMSE_metric(x_pred, x_test):
    rmse = sqrt(mean_squared_error(x_test, x_pred))
    print(rmse)
    return rmse


def EAC_metric(x_pred, x_test):
    num = np.sum(np.abs(x_pred - x_test))
    den = (np.sum(x_test))
    eac = 1 - (num/den)/2
    eac = np.where(eac < 0, 0, eac)
    print(eac)
    return eac


def NDE_metric(x_pred, x_test):
    nde = np.sum((x_test - x_pred) ** 2) / np.sum((x_test ** 2))
    print(nde)
    return nde

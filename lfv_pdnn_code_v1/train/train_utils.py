import datetime
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split

from lfv_pdnn_code_v1.common.common_utils import *

MASS_FEATURE_INDEX = 0

def prep_mass(xbtrain, xstrain, norm=None):
  np.random.seed(42)
  new = xbtrain.copy()
  total_events =len(new)
  sump = sum(xstrain[:,-1])
  import time, datetime
  start_time = time.time()
  for count, d in enumerate(new):
    mass = np.random.choice(xstrain[:,MASS_FEATURE_INDEX], p=1/sump*xstrain[:,-1])
    if norm:
      mass = mass / norm
    d[MASS_FEATURE_INDEX] = mass
    if (count % 100 == 0):
      current_time = time.time()
      speed = (count + 1.0) / (current_time - start_time)
      remaining_time = (total_events - count) / speed
      remaining_time_str = str(datetime.timedelta(seconds = (int)(remaining_time)))
      s = "%.2f" % (count*100.0/total_events)
      sys.stdout.write('\r' + s + '% events have been processed, remaining time > ' + remaining_time_str)
      sys.stdout.flush()
      #print "\r" + "events have been processed",
  print "\r100.00% events have been processed"
  return new

def prep_mass_fast(xbtrain, xstrain, mass_id = 0, norm=None):
  np.random.seed(42)
  new = xbtrain.copy()
  total_events =len(new)
  sump = sum(xstrain[:,-1])
  import time, datetime
  start_time = time.time()
  mass_list = np.random.choice(xstrain[:,mass_id], size=total_events, p=1/sump*xstrain[:,-1])
  for count, d in enumerate(new):
    if norm:
      mass = mass / norm
    d[mass_id] = mass_list[count]
    """
    if (count % 100 == 0):
      current_time = time.time()
      speed = (count + 1.0) / (current_time - start_time)
      remaining_time = (total_events - count) / speed
      remaining_time_str = str(datetime.timedelta(seconds = (int)(remaining_time)))
      s = "%.2f" % (count*100.0/total_events)
      sys.stdout.write('\r' + s + '% events have been processed, remaining time > ' + remaining_time_str)
      sys.stdout.flush()
  print "\r100.00% events have been processed"
    """
  return new

def unison_shuffled_copies(*arr):
    assert all(len(a) for a in arr)
    p = np.random.permutation(len(arr[0]))
    return (a[p] for a in arr)
    

def norweight(wt, norm=1000):
    totalWt = sum(wt)
    #print "sum of weight is: ", totalWt
    frac = norm/totalWt
    wt = frac*wt
    return wt

def get_part_feature(xtrain, nf):
    #nf = [0,1,2,3,4]
    #nf = [20,21,22,23,24,25,26,27,28,31,34,35,36,37,38,39]
    xtrain = xtrain[:,nf]
    return xtrain

def modify_array(input_array, weight_id = None, remove_negative_weight = False,
                 select_channel = False, channel_id = None, 
                 select_mass = False, mass_id = None, mass_min = None, mass_max = None,
                 reset_mass = False, reset_mass_array = None, reset_mass_id = None,
                 norm = False, sumofweight = 1000,
                 shuffle = False, shuffle_seed = 1234):
  ## modify array elements according to setup and return selected array
  ## input_array should be a numpy array
  ## select_channel: select certain channel elements from array
  ## select_mass: select elements within cerntain mass range
  ## reset_mass: random reset bkg mass with sig mass
  ## norm: normalize array weight, default sumofweight is 1000
  ## shuffle: shuffle the order of array's elements
  # copy data to avoid original data operation
  new = input_array.copy()
  # select channel
  if select_channel == True:
    if not has_none([channel_id, weight_id]):
      #print "selecting channel..."
      for ele in new:
        if ele[channel_id] != 1.0:
          ele[weight_id] = 0
    else:
      print "missing parameters, skipping channel selection..."
  # select mass range
  if select_mass == True:
    if not has_none([mass_id, mass_min, mass_max]):
      #print "selecting mass..."
      for ele in new:
        if ele[mass_id] < mass_min or ele[mass_id] > mass_max:
          ele[weight_id] = 0
    else:
      print "missing parameters, skipping mass selection..."
  # clean array
  new = clean_array(new, -1, remove_negative = remove_negative_weight, verbose = False)
  # reset mass
  if reset_mass == True:
    if not has_none([reset_mass_array, reset_mass_id]):
      #print "random reseting mass..."
      new = prep_mass_fast(new, reset_mass_array, mass_id = reset_mass_id, norm=None)
    else:
      print "missing parameters, skipping mass reset..."
  # normalize weight
  if norm == True:
    if not has_none([weight_id]):
      #print "normalizing array..."
      new[:, weight_id] = norweight(new[:, weight_id], norm = sumofweight)
    else:
      print "missing parameters, skipping normalization..."
  # shuffle array
  if shuffle == True:
    #print "shuffling array..."
    new, x2, y1, y2 = train_test_split(new, np.zeros(len(new)), test_size= 0, random_state=shuffle_seed, shuffle=True)
  # clean array
  new = clean_array(new, -1, remove_negative = remove_negative_weight, verbose = False)
  # return result
  return new

def split_and_combine(xs, xb, shuffle_before_return = True):
    ys = np.ones(len(xs))
    yb = np.zeros(len(xb))
    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size= 0.2, random_state=1234, shuffle=True)
    xb_train, xb_test, yb_train, yb_test = train_test_split(xb, yb, test_size= 0.2, random_state=1234, shuffle=True)
    x_train = np.concatenate((xs_train, xb_train))
    y_train = np.concatenate((ys_train, yb_train))
    x_test = np.concatenate((xs_test, xb_test))
    y_test = np.concatenate((ys_test, yb_test))
    # shuffle the array
    if shuffle_before_return:
        x_train, x2, y_train, y2 = train_test_split(x_train, y_train, test_size= 0, random_state=3456, shuffle=True)
        x_test, x2, y_test, y2 = train_test_split(x_test, y_test, test_size= 0, random_state=5672, shuffle=True)
    return x_train, x_test, y_train, y_test, xs_test, xb_test

def MakePlots(sig_array, bkg_array, para_index, bins=100, range=(0,100), density = False,
              xlabel="x axis", ylabel="y axis", show_plot=False, 
              save_plot = False, save_path="plots/undifined.pdf", save_format='pdf'):
  plt.hist(sig_array[:,para_index], bins=bins, weights=sig_array[:,-1], histtype='step', label='signal', range=range, density = density)
  plt.hist(bkg_array[:,para_index], bins=bins, weights=bkg_array[:,-1], histtype='step', label='background', range=range, density = density)
  plt.legend(prop={'size': 10})
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  if show_plot:
    plt.show()
  if save_plot:
    print "plotting:", save_path
    plt.savefig(save_path, format=save_format)

from ROOT import TCanvas, TColor, TGaxis, TH1F, TPad
from ROOT import kBlack, kBlue, kGreen, kRed, kViolet
from ROOT import gROOT

def CreateTH1F(array, para_index, plot_name="root plot", channel = "all", mass_min = -1, mass_max = 1e10,
               bins=100, bin_min=0, bin_max=100, xlabel="x axis", ylabel="y axis", line_color=kBlue,
               need_nor_weight = True):
    #gROOT.SetBatch(True)
    h1 = TH1F(plot_name, plot_name, bins, bin_min, bin_max)
    h1.SetLineColor(line_color)
    h1.SetLineWidth(2)

    plot_array = array * 1.0
    for element in plot_array:
      if element[para_index] == 0.0 or (element[0] < mass_min) or (element[0] > mass_max):
        element[-1] = 0
    
    if need_nor_weight:
      print "plot array normalized."
      plot_array[:,-1] = norweight(plot_array[:,-1])

    n = 0
    for element in plot_array:
      if channel == "all" and (element[-4] == 1.0 or element[-3] == 1.0 or element[-2] == 1.0):
        h1.Fill(element[para_index], element[-1])
      if channel == "emu" and element[-4] == 1.0:
        h1.Fill(element[para_index], element[-1])
      if channel == "etau" and element[-3] == 1.0:
        h1.Fill(element[para_index], element[-1])
      if channel == "mutau" and element[-2] == 1.0:
        h1.Fill(element[para_index], element[-1])
    h1.GetXaxis().SetRangeUser(bin_min, bin_max)
    h1.GetXaxis().SetTitle(xlabel)
    h1.GetXaxis().SetTitleSize(20)
    h1.GetXaxis().SetTitleFont(43)
    h1.GetXaxis().SetTitleOffset(1.55)
    h1.GetYaxis().SetTitle(ylabel)
    h1.GetYaxis().SetTitleSize(20)
    h1.GetYaxis().SetTitleFont(43)
    h1.GetYaxis().SetTitleOffset(1.55)
    h1.SetStats(0)
    return h1

def MakePlotsROOT(sig_array, bkg_array, para_index, channel = "all", 
                  plot_name="root plot", xlabel="x axis", ylabel="y axis", 
                  bins=100, x_min = 0, x_max = 1000, save_plot = False, 
                  save_path="plots/undefined.pdf", save_format='pdf', show_plot=False):
  # create histograms for signal and background
  plot_name = "(" + channel + ") " + plot_name
  hs = CreateTH1F(sig_array, para_index, plot_name=plot_name, channel = channel, 
                  bins=bins, bin_min=x_min, bin_max=x_max,
                  xlabel=xlabel, ylabel=ylabel, line_color=kRed, need_nor_weight = False)
  hb = CreateTH1F(bkg_array, para_index, plot_name=plot_name + "b", channel = channel,  
                  bins=bins, bin_min=x_min, bin_max=x_max,
                  xlabel=xlabel, ylabel=ylabel, line_color=kBlue, need_nor_weight = False)
  hs_max_bin_value = hs.GetBinContent(hs.GetMaximumBin())
  #print "max sig: ", hs_max_bin_value
  hb_max_bin_value = hb.GetBinContent(hb.GetMaximumBin())
  #print "max bkg: ", hb_max_bin_value
  hs.GetYaxis().SetRangeUser(0, 1.2 * max(hs_max_bin_value, hb_max_bin_value))
  c = TCanvas("c", "canvas", 800, 800)
  hs.Draw()
  hb.Draw("same")
  if save_plot == True:
    c.SaveAs(save_path)

def MakeComparisonPlotsROOT(first_array, second_array, para_index, channel = "all", 
                            plot_name="root plot", xlabel="x axis", ylabel="y axis", 
                            bins=100, x_min = 0, x_max = 1000, save_path="plots/undefined.pdf", 
                            save_format='pdf', show_plot=False):
  # create histograms for signal and background
  plot_name = "(" + channel + ") " + plot_name
  hs = CreateTH1F(first_array, para_index, plot_name="before DNN", channel = channel, 
                  bins=bins, bin_min=x_min, bin_max=x_max,
                  xlabel=xlabel, ylabel=ylabel, line_color=kBlue, need_nor_weight = False)
  hb = CreateTH1F(second_array, para_index, plot_name="after DNN", channel = channel,  
                  bins=bins, bin_min=x_min, bin_max=x_max,
                  xlabel=xlabel, ylabel=ylabel, line_color=kRed, need_nor_weight = False)
  hs_max_bin_value = hs.GetBinContent(hs.GetMaximumBin())
  hb_max_bin_value = hb.GetBinContent(hb.GetMaximumBin())
  # print "max bin value is:", max(hs_max_bin_value, hb_max_bin_value)
  hs.GetYaxis().SetRangeUser(0, 1.2 * max(hs_max_bin_value, hb_max_bin_value))
  c = TCanvas("c", "canvas", 800, 800)
  hs.Draw()
  hb.Draw("same")
  c.BuildLegend(0.75, 0.75, 0.9, 0.9)
  hs.SetTitle(plot_name)
  c.Draw()
  c.SaveAs(save_path)

def MakePlotsROOT_MultiChannel(sig_array_emu, sig_array_etau, sig_array_mutau, bkg_array, 
                                para_index, channel = "all", mass_min = -1, mass_max = 1e10,
                                plot_name="root plot", xlabel="x axis", ylabel="y axis", 
                                bins=100, x_min = 0, x_max = 1000, save_path="plots/undefined.pdf", 
                                save_format='pdf', show_plot=False):
  # create histograms for signal and background
  plot_name = "(all ch) " + plot_name

  hs_em = CreateTH1F(sig_array_emu, para_index, plot_name=plot_name + channel,
                      channel = "emu",
                      bins=bins, bin_min=x_min, bin_max=x_max,
                      xlabel=xlabel, ylabel=ylabel, line_color=kRed)
  hs_et = CreateTH1F(sig_array_etau, para_index, plot_name=plot_name + channel,
                      channel = "etau",
                      bins=bins, bin_min=x_min, bin_max=x_max,
                      xlabel=xlabel, ylabel=ylabel, line_color=kGreen)
  hs_mt = CreateTH1F(sig_array_mutau, para_index, plot_name=plot_name + channel,
                      channel = "mutau",
                      bins=bins, bin_min=x_min, bin_max=x_max,
                      xlabel=xlabel, ylabel=ylabel, line_color=kViolet)

  hb = CreateTH1F(bkg_array, para_index, plot_name=plot_name + "_b",
                  channel = channel, mass_min = mass_min, mass_max = mass_max,
                  bins=bins, bin_min=x_min, bin_max=x_max,
                  xlabel=xlabel, ylabel=ylabel, line_color=kBlack)

  #print "max hs_em: ", hs_em.GetBinContent(hs_em.GetMaximumBin())
  #print "max hs_et: ", hs_et.GetBinContent(hs_et.GetMaximumBin())
  #print "max hs_mt: ", hs_mt.GetBinContent(hs_mt.GetMaximumBin())

  hs_max_bin_value = max(hs_em.GetBinContent(hs_em.GetMaximumBin()),
                          hs_et.GetBinContent(hs_et.GetMaximumBin()),
                          hs_mt.GetBinContent(hs_mt.GetMaximumBin()))

  #print "hs_max_bin_value: ", hs_max_bin_value
  hb_max_bin_value = hb.GetBinContent(hb.GetMaximumBin())
  #print "hb_max_bin_value:", hb_max_bin_value
  #print "max bkg: ", hb_max_bin_value
  hs_em.GetYaxis().SetRangeUser(0, 1.2 * max(hs_max_bin_value, hb_max_bin_value))
  #print "set range user to: ", 0, ",", 1.2 * max(hs_max_bin_value, hb_max_bin_value)
  c = TCanvas("c", "canvas", 800, 800)
  hs_em.Draw()
  hs_et.Draw("same")
  hs_mt.Draw("same")
  hb.Draw("same")
  c.Draw()
  c.SaveAs(save_path)

def CalculateSignificance(xs, xb, mass_point, mass_min, mass_max, 
                          model = None, xs_model_input = None, xb_model_input = None, use_model_cut = False):
    # 1st value of each xs/xb's entry must be the mass
    signal_quantity = 0.
    background_quantity = 0.
    if (model != None) and (len(xs_model_input) != 0) and (len(xb_model_input) != 0) and (use_model_cut == True):
        signal_predict_result = model.predict(xs_model_input)
        background_predict_result = model.predict(xb_model_input)
        for n, (entry, predict_result) in enumerate(zip(xs, signal_predict_result)):
            if entry[0] > mass_min and entry[0] < mass_max and predict_result > 0.5:
                signal_quantity += entry[-1]
        for n, (entry, predict_result) in enumerate(zip(xb, background_predict_result)):
            if entry[0] > mass_min and entry[0] < mass_max and predict_result > 0.5:
                background_quantity += entry[-1]
    else:
        for entry in xs:
            if entry[0] > mass_min and entry[0] < mass_max:
                signal_quantity += entry[-1]
        for entry in xb:
            if entry[0] > mass_min and entry[0] < mass_max:
                background_quantity += entry[-1]
            
    print "for mass =", mass_point, "range = (", mass_min, mass_max, "):"
    print "  signal quantity =", signal_quantity, "background quantity =", background_quantity
    print "  significance =", signal_quantity / sqrt(background_quantity)
    
def PlotScores(sig_model_input, bkg_model_input, model, bins = 100, range = None, density = True, log = False):
  plt.hist(model.predict(sig_model_input), bins = bins, range = range, histtype='step', label='signal', density=True, log = log)
  plt.hist(model.predict(bkg_model_input), bins = bins, range = range, histtype='step', label='background', density=True, log = log)
  plt.legend(loc='upper center')
  plt.xlabel("Output score")
  plt.ylabel("arb. unit")
  plt.show()
    
def get_mass_range(mass_array, weights):
    average = np.average(mass_array, weights=weights)
    # Fast and numerically precise:
    variance = np.average((mass_array-average)**2, weights=weights)
    return np.sqrt(variance)

def plot_different_mass(mass_scan_map, input_path, para_index, model = "zprime", bins = 50, range = (-10000, 10000), 
                        density = True, xlabel="x axis", ylabel="y axis"):
    # model could be "zprime", "rpv", "qbhrs", "qbhadd"
    for i, mass in enumerate(mass_scan_map):
        # load signal
        if model == "zprime":
            xs_add = np.load(input_path + '/data_npy/emu/tree_{}00GeV.npy'.format(mass))
            """ # only use emu channel currently
            xs_temp = np.load(input_path + '/data_npy/etau/tree_{}00GeV.npy'.format(mass))
            xs_add = np.concatenate((xs_add, xs_temp))
            xs_temp = np.load(input_path + '/data_npy/mutau/tree_{}00GeV.npy'.format(mass))
            xs_add = np.concatenate((xs_add, xs_temp))
            """
        elif model == "rpv":
            xs_add = np.load(input_path + '/data_npy/emu/rpv_{}00GeV.npy'.format(mass))
            """ # only use emu channel currently
            xs_temp = np.load(input_path + '/data_npy/etau/rpv_{}00GeV.npy'.format(mass))
            xs_add = np.concatenate((xs_add, xs_temp))
            xs_temp = np.load(input_path + '/data_npy/mutau/rpv_{}00GeV.npy'.format(mass))
            xs_add = np.concatenate((xs_add, xs_temp))
            """
        xs_emu = xs_add.copy()
        # select emu channel and shuffle
        xs_emu = modify_array(xs_emu, weight_id = -1, 
                              select_channel = True, channel_id = -4,
                              norm = True, shuffle = True, shuffle_seed = 485)

        # make plots
        plt.hist(xs_emu[:, para_index], bins = bins, weights = xs_emu[:,-1], 
                 histtype='step', label='signal {}00GeV'.format(mass), range = range, density = density)
    plt.legend(prop={'size': 10})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()   
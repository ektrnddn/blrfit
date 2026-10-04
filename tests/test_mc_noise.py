"""Separate measured pixel variance from fit weights and legacy MC policy."""
import copy
import pickle
import numpy as np
import pytest

from blrfit import errors
from blrfit.model import fit as fit_module


def linear_fixture(monkeypatch):
    res=dict(wave_rest=np.arange(4.),flux_rest=np.array([10.,20.,30.,40.]),
             ivar_rest=np.array([1.,.25,0.,4.]),ivar_stat_rest=np.array([25.,16.,0.,100.]),
             z=.25,host_model=np.zeros(4),host_info={},fits={'Hbeta':{'n_broad':1}},
             settings={'mc_noise_policy':'input','fe':False,'use_ha_systemic':False},
             meas={'Hbeta':{'c50_sys':0.,'v_sys':0.}})
    seen=[]
    def continuum(w,f,ir,**kw):
        seen.append((f.copy(),ir.copy()))
        return {},np.zeros_like(f),{'solver':{'success':True}}
    def sequence(w,f,ir,cmodel,host,z,complexes,**kw):
        assert np.array_equal(ir,res['ivar_rest'])
        values={k:float(np.sum(f*ir)/np.sum(ir)) for k in errors.MC_KEYS}
        values['systemic_source']='[OIII] core'
        return {},{'Hbeta':values},{'status':'success','v_o3':0.,'snr':20.},{'Hbeta':{'status':'success'}}
    monkeypatch.setattr(errors,'fit_continuum',continuum)
    monkeypatch.setattr(fit_module,'_fit_line_sequence',sequence)
    return res,seen


@pytest.mark.parametrize('policy,key,model',[('input','ivar_stat_rest','conditional_statistical'),
    ('effective','ivar_rest','conditional_effective_noise')])
def test_noise_draws_use_selected_variance_weights_do_not_change(monkeypatch,policy,key,model):
    res,seen=linear_fixture(monkeypatch);before=pickle.dumps(res)
    mc,err,info=errors.monte_carlo(res,nmc=30,seed=481,noise_policy=policy,return_diagnostics=True)
    normals=np.random.default_rng(481).standard_normal((30,4))
    ivar=res[key];sigma=np.zeros(4);np.divide(1.,np.sqrt(ivar),out=sigma,where=ivar>0)
    expected=res['flux_rest']+normals*sigma
    for i,(f,w) in enumerate(seen):
        np.testing.assert_array_equal(f,expected[i]);np.testing.assert_array_equal(w,res['ivar_rest'])
    estimator=(expected@res['ivar_rest'])/res['ivar_rest'].sum()
    np.testing.assert_allclose(mc['Hbeta']['c50_sys'],np.percentile(estimator,[16,50,84]),atol=1e-12)
    assert err['Hbeta']['c50_sys']==pytest.approx(.5*np.diff(np.percentile(estimator,[16,84]))[0])
    assert info['uncertainty_model']==model and info['noise_variance_source']==key
    assert info['fit_weight_variance_source']=='ivar_rest' and not info['legacy_noise_policy']
    assert before==pickle.dumps(res)


def test_legacy_saved_result_retains_effective_policy(monkeypatch):
    res,seen=linear_fixture(monkeypatch);del res['settings']['mc_noise_policy'];del res['ivar_stat_rest']
    a=errors.monte_carlo(res,nmc=30,seed=731,return_diagnostics=True)
    b=errors.monte_carlo(res,nmc=30,seed=731,noise_policy='effective',return_diagnostics=True)
    assert a[:2]==b[:2]
    assert a[2]['legacy_noise_policy'] and a[2]['noise_policy']=='effective'
    assert a[2]['uncertainty_model']=='conditional_effective_noise'
    with pytest.raises(ValueError,match='requires saved'):
        errors.monte_carlo(res,nmc=30,noise_policy='input')


@pytest.mark.parametrize('bad',[[1,2],[1,np.nan,0,1],[1,-1,0,1],[1,1,1,1],[1,0,0,1]])
def test_bad_statistical_variance_refused_without_a_draw(monkeypatch,bad):
    res,seen=linear_fixture(monkeypatch);res['ivar_stat_rest']=np.array(bad)
    with pytest.raises(ValueError,match='inverse variance'):
        errors.monte_carlo(res,nmc=30)
    assert not seen


def test_missing_statistical_array_is_not_silently_floored(monkeypatch):
    res,seen=linear_fixture(monkeypatch);del res['ivar_stat_rest']
    with pytest.raises(ValueError,match='requires saved'):
        errors.monte_carlo(res,nmc=30)
    assert not seen


@pytest.mark.parametrize('policy',['unknown',None,True])
def test_bad_fit_noise_policy_rejected(policy):
    with pytest.raises(ValueError,match='mc_noise_policy'):
        fit_module.fit_spectrum([5000.],[10.],[1.],.25,mc_noise_policy=policy)


@pytest.mark.parametrize('scale',[1.,2.])
def test_statistical_variance_survives_units_frames_sorting_and_mask(monkeypatch,scale):
    monkeypatch.setattr(fit_module,'fit_continuum',lambda w,f,ir,**kw:({},np.zeros_like(f),{'solver':{'success':True}}))
    monkeypatch.setattr(fit_module,'_fit_line_sequence',lambda *a,**kw:({},{},{},{}))
    # Exact mocked dereddening transform isolates bookkeeping from its physical law.
    monkeypatch.setattr(fit_module,'deredden',lambda w,f,iv,ebv:(f*2.,iv/4.))
    res=fit_module.fit_spectrum([6000.,4000.,5000.],[10.,20.,30.],[4.,9.,0.],.25,
        ebv=.1,flux_scale=scale,host=False,fe=False,err_floor=.02)
    expected_stat=np.array([9.,0.,4.])/(scale*2.*1.25)**2
    np.testing.assert_array_equal(res['ivar_stat_rest'],expected_stat)
    expected_flux=np.array([20.,0.,10.])*scale*2.*1.25
    expected_weights=expected_stat.copy();good=expected_stat>0
    expected_weights[good]=1/(1/expected_stat[good]+(.02*expected_flux[good])**2)
    np.testing.assert_allclose(res['ivar_rest'],expected_weights,rtol=1e-14)
    np.testing.assert_array_equal(res['flux_rest'],expected_flux)
    assert res['settings']['mc_noise_policy']=='input'
    assert pickle.loads(pickle.dumps(res))['ivar_stat_rest'].tolist()==expected_stat.tolist()


def test_policy_is_exported_with_error_model(monkeypatch):
    res,_=linear_fixture(monkeypatch)
    res.update(conti={},cls={})
    res['mc'],res['err'],res['mc_info']=errors.monte_carlo(res,nmc=30,return_diagnostics=True)
    row=fit_module.summary_row(res)
    assert row['HB_mc_noise_policy']=='input'
    assert row['HB_mc_noise_variance_source']=='ivar_stat_rest'
    assert row['HB_mc_uncertainty_model']=='conditional_statistical'

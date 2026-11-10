"""Declared pilot caps, separate from validation stopping and the LR horizon."""

CAPS = {'confidence_blend': 9000, 'raw': 12000, 'gsd_blend': 9000,
        'raw_lr1e5': 12000, 'gsd_blend_lr1e5': 12000,
        'gsd_hard_readout1':9000,'raw_readout1':9000}

RISING_CAP_RULE = 'new_peak_at_cap_then_first_strict_decline_v1'


def validate_budget(budget, policy, protocol=None):
    if budget.get('rising_cap_rule') not in (None, RISING_CAP_RULE):
        raise ValueError('Unknown rising-cap rule')
    phase = budget.get('phase')
    cap = budget.get('max_updates')
    if budget.get('name') == 'dtst_recovery_budget_20261016_v1':
        variants = ('dino_recovery', 'dino_recovery_open', 'ema_recovery',
                    'dino_recovery_no_negative', 'bank_coupled', 'no_negative')
        if (phase not in variants or cap != 9000 or budget.get('original_policy') != policy or
                policy.get('mode') != 'no_improvement' or policy.get('patience_evaluations') != 2 or
                policy.get('eval_period') != 1000 or policy.get('max_updates') != 20000 or
                budget.get('user_requested') is not True or budget.get('applied_after_training_started') is not False):
            raise ValueError('Invalid semantic recovery budget')
        if protocol is not None:
            recipe = protocol['recipe']
            recovery = protocol.get('semantic_recovery')
            negative = 0 if phase in ('no_negative', 'dino_recovery_no_negative') else .5
            if (recipe['batch_size'] != 6 or recipe['base_lr'] != 2e-5 or recipe.get('readout_lr_mult') != 1 or
                    recipe['eval_period'] != 1000 or recipe['warm']['steps'] != 20000 or
                    recipe['warm']['lr_max_iter'] != 32000 or protocol['dtst']['negative_weight'] != negative or
                    protocol['source_checkpoint_sha256'] != '10fbe026b5f2dcef331a99d97bab551f7cb0a2cf95feac2625b82ca3e3449d31'):
                raise ValueError('Semantic recovery recipe drift')
            if phase in ('bank_coupled', 'no_negative'):
                if recovery is not None: raise ValueError('Control unexpectedly enables recovery')
            elif (recovery is None or recovery['features'] != ('ema' if phase == 'ema_recovery' else 'dinov2') or
                    recovery['max_native_confidence'] != (1. if phase == 'dino_recovery_open' else .95)):
                raise ValueError('Wrong semantic recovery arm')
            if phase == 'bank_coupled':
                if (protocol['geometry_study']['donor_mode'] != 'coupled' or
                        protocol['gsd_study']['scout']['variant'] != 'source_bank'):
                    raise ValueError('Wrong bank/geometry combination')
            elif protocol['gsd_study']['arm'] != 'raw':
                raise ValueError('Recovery arms must preserve raw DT-ST supervision')
        return cap
    if budget.get('name') == 'dtst_pheno_budget_20261031_v1':
        arms = {'pheno_c0': (True, False), 'pheno_c1': (False, True), 'pheno_c0_c1': (True, True), 'pheno_null': (False, False),
                'pheno_c0_protect': (True, False), 'pheno_delta_replay': (False, False), 'pheno_c0_c1_randdir': (True, True),
                # 09/11/2026 IRRG->RGB study (per-class G for C0, per-pixel plausible C1); same rule and cap
                'pheno_c0multi': (True, False), 'pheno_c0multi_c1plaus': (True, True), 'pheno_c0_c1plaus': (True, True)}
        if (phase not in arms or cap != 9000 or budget.get('original_policy') != policy or
                policy.get('mode') != 'no_improvement' or policy.get('patience_evaluations') != 2 or
                policy.get('eval_period') != 1000 or policy.get('max_updates') != 20000 or
                budget.get('user_requested') is not True or budget.get('applied_after_training_started') is not False):
            raise ValueError('Invalid phenology budget')
        if protocol is not None:
            recipe, pheno = protocol['recipe'], protocol.get('phenology')
            if (pheno is None or (pheno['c0'], pheno['c1']) != arms[phase] or protocol['gsd_study']['arm'] != 'raw' or
                    recipe['batch_size'] != 6 or recipe['base_lr'] != 2e-5 or recipe.get('readout_lr_mult') != 1 or
                    recipe['eval_period'] != 1000 or recipe['warm']['steps'] != 20000 or
                    recipe['warm']['lr_max_iter'] != 32000 or protocol['dtst']['negative_weight'] != .5 or
                    protocol['source_checkpoint_sha256'] != '10fbe026b5f2dcef331a99d97bab551f7cb0a2cf95feac2625b82ca3e3449d31'):
                raise ValueError('Phenology protocol drift')
        return cap
    if budget.get('name') == 'dtst_geometry_budget_20261016_v1':
        variants = {'coupled_scale': ('coupled', False), 'independent_scale': ('independent', False),
                    'masked_context': ('baseline', True)}
        if (phase not in variants or cap != 9000 or budget.get('original_policy') != policy or
                policy.get('mode') != 'no_improvement' or policy.get('patience_evaluations') != 2 or
                policy.get('eval_period') != 1000 or policy.get('max_updates') != 20000 or
                budget.get('user_requested') is not True or budget.get('applied_after_training_started') is not False):
            raise ValueError('Invalid geometry budget')
        if protocol is not None:
            recipe, geometry = protocol['recipe'], protocol['geometry_study']
            mode, context = variants[phase]
            if (geometry['donor_mode'] != mode or (geometry.get('context') is not None) != context or
                    protocol['gsd_study']['arm'] != 'raw' or recipe['batch_size'] != 6 or
                    recipe['base_lr'] != 2e-5 or recipe.get('readout_lr_mult') != 1 or
                    recipe['eval_period'] != 1000 or recipe['warm']['steps'] != 20000 or
                    recipe['warm']['lr_max_iter'] != 32000 or recipe['snd']['steps'] != 12000 or
                    protocol['source_checkpoint_sha256'] != '10fbe026b5f2dcef331a99d97bab551f7cb0a2cf95feac2625b82ca3e3449d31'):
                raise ValueError('Geometry protocol drift')
        return cap
    if budget.get('name') == 'dtst_scout_budget_20261015_v1':
        variants = {'support_hard':'gsd', 'source_veto':'confidence', 'source_bank':'confidence',
                    'support_bank':'gsd', 'support_fusion_bank':'gsd', 'generic_support_bank':'generic',
                    'confidence_support_bank':'confidence', 'raw_readout1_scout':'raw'}
        if (phase not in variants or cap != 9000 or budget.get('original_policy') != policy or
                policy.get('mode') != 'no_improvement' or policy['patience_evaluations'] != 2 or
                policy['eval_period'] != 1000 or policy['max_updates'] != 20000 or
                not budget.get('user_requested') or budget.get('applied_after_training_started') is not False):
            raise ValueError('Invalid scout budget')
        if protocol is not None:
            recipe = protocol['recipe']
            if (protocol['gsd_study']['arm'] != variants[phase] or recipe['batch_size'] != 6 or
                    recipe['base_lr'] != 2e-5 or recipe.get('readout_lr_mult') != 1 or
                    recipe['eval_period'] != 1000 or recipe['warm']['steps'] != 20000 or
                    recipe['warm']['lr_max_iter'] != 32000 or recipe['snd']['steps'] != 12000 or
                    protocol['source_checkpoint_sha256'] != '10fbe026b5f2dcef331a99d97bab551f7cb0a2cf95feac2625b82ca3e3449d31'):
                raise ValueError('Scout protocol drift')
        return cap
    if (not isinstance(policy, dict) or budget.get('name') != 'dtst_pilot_caps_20261015_v1' or phase not in CAPS or
            cap != CAPS[phase] or budget.get('original_policy') != policy or
            policy['max_updates'] != 20000 or policy['eval_period'] != 1000 or
            not policy['min_updates'] <= cap < policy['max_updates'] or
            cap % policy['eval_period'] or not budget.get('user_requested') or
            not isinstance(budget.get('applied_after_training_started'), bool)):
        raise ValueError('Invalid declared pilot budget')
    if protocol is not None:
        recipe = protocol['recipe']
        arm = 'confidence' if phase == 'confidence_blend' else ('raw' if phase.startswith('raw') else 'gsd')
        lr = 1e-5 if phase.endswith('lr1e5') else 2e-5
        if (protocol['gsd_study']['arm'] != arm or recipe['batch_size'] != 6 or
                recipe['base_lr'] != lr or recipe['eval_period'] != 1000 or
                recipe['warm']['steps'] != 20000 or recipe['warm']['lr_max_iter'] != 32000):
            raise ValueError('Pilot budget does not match the run recipe')
        if phase.endswith('readout1') and recipe.get('readout_lr_mult',10)!=1:
            raise ValueError('Priority pilot must use the declared readout multiplier1')
    return cap


def cap_extension_active(curve, budget, policy):
    cap = validate_budget(budget, policy)
    if budget.get('rising_cap_rule') != RISING_CAP_RULE:
        return False
    at_cap = next((p for p in curve if p['iteration'] == cap), None)
    before = [p['metrics']['miou'] for p in curve if p['iteration'] < cap]
    return bool(at_cap and before and at_cap['metrics']['miou'] > max(before))


def cap_decision(curve, budget, policy):
    cap = validate_budget(budget, policy)
    if not curve or curve[-1]['iteration'] < cap:
        return None
    if cap_extension_active(curve, budget, policy):
        if curve[-1]['iteration'] >= policy['max_updates']:
            return None
        peak = max(curve, key=lambda point: point['metrics']['miou'])
        if curve[-1]['metrics']['miou'] >= peak['metrics']['miou']:
            return None
        return dict(iteration=curve[-1]['iteration'], peak_iteration=peak['iteration'],
                    peak_miou=peak['metrics']['miou'], recent_miou=[curve[-1]['metrics']['miou']],
                    pilot_cap_extension=True, pilot_budget=budget,
                    reason='New peak at pilot cap; first subsequent validation strictly below peak')
    if curve[-1]['iteration'] != cap:
        raise RuntimeError('Saved pilot cap checkpoint window was missed')
    peak = max(curve, key=lambda point: point['metrics']['miou'])
    return dict(iteration=cap, peak_iteration=peak['iteration'], peak_miou=peak['metrics']['miou'],
                recent_miou=[curve[-1]['metrics']['miou']], pilot_budget_cap=True,
                pilot_budget=budget, reason='User-declared pilot budget reached; convergence not claimed')


def effective_policy(policy, budget):
    validate_budget(budget, policy)
    result = dict(policy, pilot_max_updates=budget['max_updates'],
                  pilot_budget_name=budget['name'])
    if budget.get('rising_cap_rule'):
        result['rising_cap_rule'] = budget['rising_cap_rule']
    return result

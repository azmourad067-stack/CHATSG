-- Rôle de l'application Streamlit : lecture seule. La connexion (LOGIN + mot de passe)
-- est activée par le propriétaire du projet, pour qu'aucun secret ne transite ailleurs.
do $$ begin
  if not exists (select from pg_roles where rolname = 'hi_app') then
    create role hi_app nologin;
  end if;
end $$;
grant usage on schema public to hi_app;

-- Supabase expose le schéma public via son API REST (rôles anon / authenticated) :
-- RLS activé partout, droits retirés à ces rôles, lecture accordée au seul rôle hi_app.
do $$
declare t text;
begin
  foreach t in array array[
    'collection_runs','data_quality_reports','data_sources','detected_patterns','feature_sets',
    'horses','jockeys','owners','racecourses','raw_payloads','research_documents','stored_files',
    'trainers','model_versions','races','weather_conditions','model_metrics','odds_snapshots',
    'payouts','predictions','race_entries','race_results','racecourse_conditions',
    'prediction_entries','prediction_explanations']
  loop
    execute format('alter table public.%I enable row level security', t);
    execute format('revoke all on table public.%I from anon, authenticated', t);
    execute format('grant select on table public.%I to hi_app', t);
    execute format('drop policy if exists hi_app_read on public.%I', t);
    execute format('create policy hi_app_read on public.%I for select to hi_app using (true)', t);
  end loop;
end $$;
revoke all on all sequences in schema public from anon, authenticated;
-- tables créées plus tard (évolutions du schéma) : jamais exposées à l'API publique
alter default privileges in schema public revoke all on tables from anon, authenticated;
alter default privileges in schema public revoke all on sequences from anon, authenticated;

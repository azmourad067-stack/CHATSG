-- La collecte (rôle hi_writer, secret GitHub DATABASE_URL) écrit les prédictions du jour :
-- mêmes droits et même politique que sur les autres tables qu'elle remplit.
do $$
declare t text;
begin
  foreach t in array array['day_prediction_races', 'day_prediction_runners']
  loop
    execute format('grant select, insert, update, delete on table public.%I to hi_writer', t);
    execute format('drop policy if exists hi_writer_all on public.%I', t);
    execute format('create policy hi_writer_all on public.%I for all to hi_writer '
                   'using (true) with check (true)', t);
  end loop;
end $$;

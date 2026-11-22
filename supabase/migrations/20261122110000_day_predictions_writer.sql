-- La collecte (rôle hi_writer, secret GitHub DATABASE_URL) écrit les prédictions du jour :
-- mêmes droits et même politique que sur les autres tables qu'elle remplit.
-- (Appliqué tel quel le 22/11/2026, sans « drop policy » qui demande une confirmation.)
grant select, insert, update, delete on table public.day_prediction_races to hi_writer;
grant select, insert, update, delete on table public.day_prediction_runners to hi_writer;
create policy hi_writer_all on public.day_prediction_races for all to hi_writer
    using (true) with check (true);
create policy hi_writer_all on public.day_prediction_runners for all to hi_writer
    using (true) with check (true);

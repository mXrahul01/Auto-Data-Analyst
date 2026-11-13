import { useEffect, useState } from 'react';
import { getCatalog, type Catalog } from './api/client';

const roadmap = [
  ['01', 'Import and inspect', 'Bounded CSV upload, schema confirmation, immutable originals.'],
  ['02', 'Understand and prepare', 'Quality evidence, reviewed transformations, dataset versions.'],
  ['03', 'Explore and train', 'Consistent dashboards and reproducible model evaluation.'],
];

export default function App() {
  const [catalog, setCatalog] = useState<Catalog>();
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setCatalog(undefined);
    setError('');
    getCatalog(controller.signal).then(setCatalog).catch((e: unknown) => {
      if (!controller.signal.aborted) setError(e instanceof Error ? e.message : 'Request failed.');
    });
    return () => controller.abort();
  }, [attempt]);
  return <main>
    <header><a className="wordmark" href="/">ADA<span> / Auto Data Analyst</span></a><span className="tag">Development foundation</span></header>
    <section className="hero"><p className="eyebrow">DATA WORKSPACE · MICROSERVICES</p>
      <h1>From raw data<br/>to clear decisions.</h1>
      <p className="intro">A foundation for traceable data preparation, interactive analysis, and reproducible machine learning.</p>
      <p className="notice">Scaffold stage: service health and this status page are available. Upload, cleaning, dashboards, and training are planned.</p>
    </section>
    <section aria-labelledby="status-title"><div className="section-heading"><div><p className="eyebrow">LOCAL ENVIRONMENT</p><h2 id="status-title">Service connections</h2></div><button onClick={() => setAttempt(n => n + 1)}>Refresh status</button></div>
      <p className="muted">Reachability checks verify HTTP bootstrap endpoints. They do not verify data or ML features.</p>
      <div aria-live="polite">{error ? <p role="alert" className="error">{error}</p> : !catalog ? <p>Checking gateway and services…</p> :
        <div className="grid">{catalog.services.map(s => <article className="service" key={s.name}><span className={'state ' + s.state}>{s.state === 'reachable' ? 'Reachable' : 'Unavailable'}</span><h3>{s.name}</h3><code>{s.health_path}</code><p className="muted">HTTP bootstrap · {s.stage}</p></article>)}</div>}
      </div>
    </section>
    <section className="roadmap" aria-labelledby="roadmap-title"><p className="eyebrow">NEXT MILESTONES</p><h2 id="roadmap-title">Build one complete workflow at a time.</h2><div className="grid">{roadmap.map(([n,title,text]) => <article key={n}><span className="step">{n}</span><h3>{title}</h3><p>{text}</p><small>Planned</small></article>)}</div></section>
    <footer>Auto Data Analyst <span>Foundation scaffold · local development only</span></footer>
  </main>;
}

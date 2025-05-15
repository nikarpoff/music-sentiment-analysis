import Home from './components/home/Home';
import classes from './app.module.css'

function App() {
    return (
        <div className={classes.app}>
            <header>
                <h1 className={classes.header}>
                    Определение эмоциональной окраски музыкальных произведений
                </h1>
            </header>
            
            <main>
                <Home />
            </main>
        </div>
    )
}

export default App;

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { supabase } from './supabaseClient';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isSignup, setIsSignup] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  const [darkMode, setDarkMode] = useState(false);
  const [patientName, setPatientName] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('screening'); // screening or history

  useEffect(() => {
    // Restore an existing Supabase session on page load/refresh
    supabase.auth.getSession().then(({ data: { session } }) => {
      setIsLoggedIn(!!session);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsLoggedIn(!!session);
    });

    return () => subscription.unsubscribe();
  }, []);

  useEffect(() => {
    if (isLoggedIn) {
      fetchHistory();
    }
  }, [isLoggedIn]);

  const getAuthHeaders = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    return session ? { Authorization: `Bearer ${session.access_token}` } : {};
  };

  const fetchHistory = async () => {
    try {
      const headers = await getAuthHeaders();
      const res = await axios.get(`${API_BASE_URL}/api/history`, { headers });
      setHistory(res.data);
    } catch (err) {
      console.error("Error fetching history", err);
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setAuthError('');

    if (!email || !password) {
      setAuthError('Enter both email and password.');
      return;
    }

    setAuthLoading(true);
    try {
      if (isSignup) {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        setAuthError('Account created. Check your email to confirm, then log in.');
        setIsSignup(false);
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        // onAuthStateChange listener above sets isLoggedIn automatically
      }
    } catch (err) {
      setAuthError(err.message || 'Authentication failed.');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("patient_name", patientName || "Walk-in Patient");
    formData.append("language", "english");

    setLoading(true);
    try {
      const headers = await getAuthHeaders();
      const response = await axios.post(`${API_BASE_URL}/api/screen`, formData, {
        headers: { "Content-Type": "multipart/form-data", ...headers }
      });
      setResult(response.data);
      fetchHistory();
    } catch (err) {
      console.error(err);
      alert("Error processing image. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  const downloadPDF = () => {
    window.print();
  };

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
        <div className="bg-white p-8 rounded-2xl shadow-xl w-full max-w-md">
          <h1 className="text-2xl font-bold text-blue-600 mb-2 text-center">👁️ RetinaScan AI</h1>
          <p className="text-xs text-center text-slate-400 mb-6">Build by Malik Junaid Ahmad</p>

          <form onSubmit={handleAuth} className="space-y-4">
            <div>
              <label className="text-xs font-semibold text-slate-600">Email Address</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full mt-1 p-3 border rounded-xl text-sm focus:outline-blue-500"
                placeholder="doctor@hospital.com"
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-600">Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full mt-1 p-3 border rounded-xl text-sm focus:outline-blue-500"
                placeholder="••••••••"
              />
            </div>
            <button type="submit" disabled={authLoading} className="w-full bg-blue-600 text-white p-3 rounded-xl font-semibold hover:bg-blue-700 transition disabled:bg-slate-400">
              {authLoading ? "Please wait..." : (isSignup ? "Create Account & Login" : "Clinical Portal Login")}
            </button>
          </form>

          {authError && (
            <p className="mt-3 text-xs text-center text-rose-600 font-medium">{authError}</p>
          )}

          <button
            onClick={() => setIsSignup(!isSignup)}
            className="w-full mt-4 text-xs text-blue-600 font-medium hover:underline text-center"
          >
            {isSignup ? "Already have an account? Login" : "New clinician? Register Account"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen transition-colors duration-200 ${darkMode ? 'bg-slate-900 text-slate-100' : 'bg-slate-50 text-slate-800'} font-sans p-6`}>
      <header className={`max-w-6xl mx-auto mb-8 flex justify-between items-center border-b pb-4 ${darkMode ? 'border-slate-800' : 'border-slate-200'}`}>
        <div>
          <h1 className="text-2xl font-bold text-blue-500 flex items-center gap-2">
            👁️ RetinaScan AI
          </h1>
          <p className="text-xs text-slate-400">Explainable DR Screening • Build by <span className="font-semibold text-blue-400">Malik Junaid Ahmad</span></p>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setActiveTab(activeTab === 'screening' ? 'history' : 'screening')}
            className={`text-xs px-3 py-1.5 rounded-lg font-medium border ${darkMode ? 'border-slate-700 bg-slate-800 text-slate-200' : 'bg-white text-slate-700'}`}
          >
            {activeTab === 'screening' ? '📊 Patient History Log' : '🔬 Screening Dashboard'}
          </button>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 rounded-xl border ${darkMode ? 'border-slate-700 bg-slate-800 text-amber-400' : 'bg-white text-slate-600'}`}
          >
            {darkMode ? '☀️ Light' : '🌙 Dark'}
          </button>
          <button
            onClick={handleLogout}
            className="text-xs bg-rose-50 text-rose-600 border border-rose-200 px-3 py-1.5 rounded-lg font-medium hover:bg-rose-100"
          >
            Logout
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto">
        {activeTab === 'history' ? (
          <div className={`p-6 rounded-2xl border shadow-sm ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
            <h2 className="text-lg font-semibold mb-4">Patient Longitudinal Screening History</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className={`border-b text-xs uppercase ${darkMode ? 'border-slate-700 text-slate-400' : 'border-slate-200 text-slate-500'}`}>
                  <tr>
                    <th className="pb-3">Timestamp</th>
                    <th className="pb-3">Patient Name</th>
                    <th className="pb-3">Stage Result</th>
                    <th className="pb-3">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/20">
                  {history.map((item) => (
                    <tr key={item.id} className="hover:bg-slate-500/5">
                      <td className="py-3 text-xs text-slate-400">{item.timestamp}</td>
                      <td className="py-3 font-medium">{item.patient_name}</td>
                      <td className="py-3"><span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-md font-semibold">{item.prediction}</span></td>
                      <td className="py-3 font-bold">{item.confidence}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Upload & Preview Card */}
            <div className={`p-6 rounded-2xl border shadow-sm flex flex-col justify-between ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
              <div>
                <h2 className="text-lg font-semibold mb-4">1. Scan Input & Patient Details</h2>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label className="text-xs font-semibold text-slate-400">Patient Full Name / ID</label>
                    <input
                      type="text"
                      value={patientName}
                      onChange={(e) => setPatientName(e.target.value)}
                      placeholder="e.g. Muhammad Ali"
                      className={`w-full mt-1 p-2.5 rounded-xl border text-sm ${darkMode ? 'bg-slate-900 border-slate-700 text-white' : 'bg-slate-50 border-slate-200'}`}
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-400">Retinal Fundus Image (.jpg / .png)</label>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageChange}
                      className="block w-full mt-1 text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
                    />
                  </div>

                  {preview && (
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div>
                        <p className="text-xs text-slate-400 mb-1 font-medium">Original Scan</p>
                        <img src={preview} alt="Retina Preview" className="w-full h-44 object-cover rounded-xl border" />
                      </div>
                      {result && (
                        <div>
                          <p className="text-xs text-slate-400 mb-1 font-medium">Grad-CAM Heatmap</p>
                          <img src={result.gradcam_image} alt="GradCAM" className="w-full h-44 object-cover rounded-xl border" />
                        </div>
                      )}
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={!selectedFile || loading}
                    className="w-full mt-4 bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition disabled:bg-slate-600"
                  >
                    {loading ? "Analyzing Scan & Generating Reports..." : "Run AI Diagnostic Screening"}
                  </button>
                </form>
              </div>

              {result && (
                <div className={`mt-6 p-4 rounded-xl border ${darkMode ? 'bg-slate-900 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Classification:</span>
                    <span className="text-sm font-bold text-blue-500">{result.prediction}</span>
                  </div>
                  <div className="flex justify-between items-center mt-2">
                    <span className="text-sm font-medium">Confidence Score:</span>
                    <span className="text-sm font-bold">{result.confidence}%</span>
                  </div>
                  {result.warning_flag && (
                    <div className="mt-3 p-2 bg-amber-500/10 border border-amber-500/30 text-amber-500 text-xs rounded-lg font-medium">
                      ⚠️ Low Confidence Warning: Manual Ophthalmologist Review Required.
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Generative Reports Dashboard */}
            <div className={`p-6 rounded-2xl border shadow-sm flex flex-col justify-between ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">2. Generative Clinical Reports</h2>
                  {result && (
                    <button
                      onClick={downloadPDF}
                      className="text-xs bg-emerald-600 text-white px-3 py-1.5 rounded-lg font-semibold hover:bg-emerald-700 transition"
                    >
                      🖨️ Download / Print Report
                    </button>
                  )}
                </div>

                {!result ? (
                  <div className="h-72 flex flex-col items-center justify-center text-slate-400 text-center border-2 border-dashed rounded-xl border-slate-700/30">
                    <p>Upload a fundus image and run analysis</p>
                    <p className="text-xs mt-1">Multi-audience narratives & Urdu summaries will appear here</p>
                  </div>
                ) : (
                  <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-wider text-blue-500 mb-1">Physician Summary</h3>
                      <div className={`p-3 rounded-xl border text-xs leading-relaxed whitespace-pre-wrap ${darkMode ? 'bg-slate-900/50 border-slate-700 text-slate-300' : 'bg-blue-50/50 border-blue-100 text-slate-700'}`}>
                        {result.clinical_report}
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-wider text-emerald-500 mb-1">Patient Plain-Language (English)</h3>
                      <div className={`p-3 rounded-xl border text-xs leading-relaxed whitespace-pre-wrap ${darkMode ? 'bg-slate-900/50 border-slate-700 text-slate-300' : 'bg-emerald-50/50 border-emerald-100 text-slate-700'}`}>
                        {result.patient_report}
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-wider text-purple-500 mb-1">Patient Plain-Language (Urdu / اردو)</h3>
                      <div className={`p-3 rounded-xl border text-xs leading-relaxed whitespace-pre-wrap font-urdu ${darkMode ? 'bg-slate-900/50 border-slate-700 text-slate-300' : 'bg-purple-50/50 border-purple-100 text-slate-700'}`}>
                        {result.urdu_report}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-6 text-xs text-slate-400 text-center border-t border-slate-700/20 pt-4">
                RetinaScan AI • Architected by Malik Junaid Ahmad for Hackathon Excellence
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

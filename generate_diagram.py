import graphviz

# Define NeurIPS-friendly colors (colorblind-friendly palette)
colors = {
    'bg': '#FFFFFF',
    'node_bg': '#FDFBF7',     # Warm white for nodes
    'edge': '#333333',        # Dark gray for edges
    'stage1': '#E69F00',      # Orange
    'stage2': '#56B4E9',      # Sky Blue
    'stage3': '#009E73',      # Bluish Green
    'stage4': '#0072B2',      # Blue
    'stage5': '#D55E00',      # Vermillion
    'font': '#222222',
    'sub_bg': '#F4F4F4'
}

dot = graphviz.Digraph('SWA_MPPI', format='pdf', engine='dot')
dot.attr(rankdir='TB', size='10,8', dpi='300', fontname='Times New Roman', fontcolor=colors['font'], bgcolor=colors['bg'])
dot.attr('node', shape='box', style='rounded,filled', fontname='Times New Roman', fillcolor=colors['node_bg'], color=colors['edge'], penwidth='1.5', margin='0.3,0.1')
dot.attr('edge', color=colors['edge'], penwidth='1.5', fontname='Times New Roman', arrowsize='0.8')

# Global Title
dot.attr(label='SWA-MPPI Pipeline: Dynamic Social Consensus for Cross-Cultural Value Negotiation\n', labelloc='t', fontsize='18', fontname='Times New Roman-Bold')

# Subgraph: Input Data
with dot.subgraph(name='cluster_0') as c:
    c.attr(label='1. Cultural & Scenario Grounding', style='rounded', color=colors['stage1'], bgcolor='#FFF9ED', penwidth='2')
    c.node('WVS', 'World Values Survey (WVS)\n[Cross-National Wave 7]', shape='cylinder')
    c.node('MultiTP', 'Moral Machine Scenarios\n(MultiTP / Synthesized)', shape='note')
    c.node('Persona', 'Target Country Personas\n(e.g., USA, CHN, JPN, BRA)')
    
# Subgraph: Multilingual Alignment (SWA)
with dot.subgraph(name='cluster_1') as c:
    c.attr(label='2. Systematic Worldview Alignment (SWA)', style='rounded', color=colors['stage2'], bgcolor='#EDF6FD', penwidth='2')
    c.node('NativeLang', 'Native Language Translation\n(Characters, Scenarios, Prompts)')
    c.node('PersonaInject', 'Implicit Pre-Logit Control\n[Context Injection]')
    c.edge('NativeLang', 'PersonaInject')

# Subgraph: MPPI Core
with dot.subgraph(name='cluster_2') as c:
    c.attr(label='3. Model Predictive Path Integral (MPPI)', style='rounded', color=colors['stage3'], bgcolor='#EDF8F4', penwidth='2')
    c.node('Sampling', 'Action Trajectory Sampling\n(K=128 samples with Noise Std = 0.3)', shape='parallelogram')
    c.node('Prospect', 'Prospect Theory Value Function\n(Diminishing Sensitivity α, β & Loss Aversion κ)')
    c.node('Cost', 'Dynamic Cost Aggregation\n(KL Constraint & Cooperative Target)')
    c.edge('Sampling', 'Prospect')
    c.edge('Prospect', 'Cost')

# Subgraph: Output
with dot.subgraph(name='cluster_3') as c:
    c.attr(label='4. Output & Decision', style='rounded', color=colors['stage4'], bgcolor='#EEF4F9', penwidth='2')
    c.node('Decision', 'Decision Sharpening\n(Temperature Scaling)', shape='rarrow')
    c.node('Output', 'Final Moral Consensus\n(LEFT vs RIGHT)', shape='ellipse', fillcolor=colors['stage5'], fontcolor='white', color=colors['stage5'])
    c.edge('Decision', 'Output')

# Connecting main branches
dot.edge('WVS', 'Persona')
dot.edge('Persona', 'NativeLang')
dot.edge('MultiTP', 'NativeLang', ' Dilemma Input')
dot.edge('PersonaInject', 'Sampling', ' LLM Logits')
dot.edge('Cost', 'Decision', ' Optimal Path')

# Save as PDF and PNG
dot.render('swa_mppi_pipeline_overview', format='pdf', cleanup=True)
dot.render('swa_mppi_pipeline_overview', format='png', cleanup=True)
print("Saved diagram to swa_mppi_pipeline_overview.pdf and .png")
